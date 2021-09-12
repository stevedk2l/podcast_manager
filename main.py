import argparse
import atexit
import json
import logging
import os
import pickle
import shutil
import ssl
import tempfile
import threading
import time
import urllib
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from functools import lru_cache
from logging import config
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from xml.etree import ElementTree

import certifi
import ffmpeg
import podcastparser
import pymediainfo
import requests
import yaml
from box import Box
from tqdm import tqdm

log_config = {
    "version": 1,
    "root": {
        "handlers": ["console"],
        "level": "DEBUG"
    },
    "handlers": {
        "console": {
            "formatter": "std_out",
            "class": "logging.StreamHandler",
            "level": "DEBUG"
        }
    },
    "formatters": {
        "std_out": {
            "format": "%(asctime)s.%(msecs)03d  [%(levelname)s] %(module)s:%(lineno)d %(threadName)s)) -- %(message)s",
            "datefmt": "%d-%m-%Y %H:%M:%S"
        }
    },
}

config.dictConfig(log_config)

CONFIG: Box
ROOT_PATH: str = ""
CLEANUPS = []
CLEANUPS_LOCK: threading.Lock = threading.Lock()

format_rectifier = {"opus": "libopus"}


# https://stackoverflow.com/a/64030200/10619293
def retry(times, exceptions):
    """
    Retry Decorator
    Retries the wrapped function/method `times` times if the exceptions listed
    in ``exceptions`` are thrown
    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    :param Exceptions: Lists of exceptions that trigger a retry attempt
    :type Exceptions: Tuple of Exceptions
    """

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    logging.warning(
                        'Exception thrown when attempting to run %s, attempt '
                        '%d of %d' % (func, attempt, times)
                    )
                    attempt += 1
            return func(*args, **kwargs)

        return newfn

    return decorator


class PodcastSet(set):
    def __getitem__(self, item):
        for ele in self:
            if ele == item:
                return ele

    def __contains__(self, item):
        if isinstance(item, str):
            for ele in self:
                if ele.title == item:
                    return True
            return False
        else:
            return super().__contains__(item)


class Podcast:
    instances = PodcastSet()

    def __init__(self, feed):
        self.feed = feed
        pod = self.fetch_feed()

        self.title: str = pod["title"].replace(" ", "_")
        if self in Podcast.instances:
            self.episodes = Podcast.instances[self].episodes
            self.update_episodes(pod["episodes"])
            self.instances.remove(self)  # Gets rid of the original instance with this same name
            self.instances.add(self)  # Add this new instance
        else:
            self.episodes: List[PodcastEpisode] = self.create_episodes(pod["episodes"])
            self.instances.add(self)
        logging.debug(f"Imported podcast: {self.title}")

    def create_episodes(self, parsed_feed) -> List['PodcastEpisode']:
        eps = []
        for ep in parsed_feed:
            eps.append(PodcastEpisode(self, ep))
        return eps

    def update_episodes(self):
        for x in range(len(self.episodes), len(parsed_feed := self.fetch_feed()['episodes'])):
            self.episodes.append(PodcastEpisode(self, parsed_feed[x]))

    def process_episodes(self):
        for ep in self.episodes:
            logging.debug(f"Processing: {self.title} -- {ep.name}")
            ep.do_it()

    @staticmethod
    def pickle_it():
        with open('pickled.cache', 'wb') as f:
            pickle.dump(Podcast.instances, f)

    @staticmethod
    def unpickle_it(path: str = 'pickled.cache'):
        with open(path, 'rb') as f:
            try:
                return pickle.load(f)
            except Exception as e:
                logging.exception(e)
            return []

    @retry(times=5, exceptions=(URLError, ConnectionRefusedError,))
    def fetch_feed(self):
        with urllib.request.urlopen(self.feed, context=ssl.create_default_context(cafile=certifi.where())) as response:
            pod = podcastparser.parse(self.feed, response)
            return pod

    def __str__(self):
        return self.title

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.title == other.title

    def __hash__(self):
        return hash(self.title)


UNITS = {"B": 1, "K": 1024, "M": 1048576}


def parse_size(size) -> int:
    unit = size[-1].upper()
    number = size[:-1]
    return int(float(number) * UNITS[unit])


@lru_cache(maxsize=1)
def get_req_info() -> Tuple[str, int]:
    return CONFIG.format, parse_size(CONFIG.bitrate)


def record_path_for_delete(f):
    def wrapper():
        global CLEANUPS
        to_be_killed_file = f()
        with CLEANUPS_LOCK:
            CLEANUPS.append(to_be_killed_file)
        return to_be_killed_file

    return wrapper


def cleanup():
    global CLEANUPS
    for path in CLEANUPS:
        try:
            os.remove(path)
            logging.debug(f"Deleted: {path}")
        except:
            pass


@record_path_for_delete
def get_scratch_file():
    return os.path.join(tempfile.gettempdir(), f"{uuid.uuid1()}")


class PodcastEpisode:
    def __init__(self, podcast, ep):
        self.podcast: Podcast = podcast
        self.name: str = ep['title']
        self.released: int = ep['published']
        try:
            self.url: str = ep['enclosures'][0]['url']
        except IndexError:
            try:
                self.url: str = ep["link"]
            except Exception as e:
                raise e
        self.tempfile: str = get_scratch_file()
        self.guid: str = ep['guid']
        self._requested_format, self._requested_bitrate = get_req_info()

        self.conversion_filepath = os.path.join("/tmp", f"{self.guid}.{self._requested_format}")
        self.filepath = os.path.join(ROOT_PATH,
                                     self.podcast.title,
                                     f"{self.podcast.title}__{self.released}__{self.name}.{self._requested_format}")

        self.completed = os.path.exists(self.filepath)

        if self.completed:
            self.format, self.bitrate = self.collect_audio_metadata()
            req_format, req_bitrate = get_req_info()
            if not ((self.format.lower() == req_format.lower()) or (
                    req_bitrate * 0.9 <= self.bitrate <= req_bitrate * 1.1)):
                self.completed = False
                logging.info(f"Marking: {self} as not complete")
        else:
            self.format, self.bitrate = None, None

    def __hash__(self):
        return hash(self.guid)

    def __repr__(self):
        return f"{self.podcast.title} -- {self.name}"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def decode(stuff) -> Union['PodcastEpisode', List['PodcastEpisode']]:
        container = json.loads(stuff)
        if isinstance(container, list):
            logging.info(msg=f"Importing: {len(container)} podcasts to memory")
            return [PodcastEpisode(**deserial) for deserial in container]
        else:
            logging.info(msg="Importing 1 podcast to memory")
            return PodcastEpisode(**container)

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    def _create_folder(self):
        folder = ROOT_PATH + self.podcast.title
        if not os.path.exists(folder):
            logging.debug(f"Creating folder: {folder}")
            os.mkdir(folder)

    def convert(self):
        logging.debug(msg=f"Converting episode: {self}")
        ffmpeg.input(self.tempfile).output(self.conversion_filepath,
                                           format=self._requested_format,
                                           acodec=format_rectifier[self._requested_format],
                                           ac=1,
                                           audio_bitrate=self._requested_bitrate) \
            .run(capture_stdout=True, capture_stderr=True)
        logging.debug(msg=f"Converted episode: {self}")

        os.remove(self.tempfile)
        logging.debug("Removed the downloaded temp file")

        shutil.move(self.conversion_filepath, self.filepath)
        logging.debug(f"Moved the converted file to: {self.filepath}")

        self.format = self._requested_format
        self.bitrate = self._requested_bitrate
        self._requested_format, self._requested_bitrate = None, None

    def cleanup_temps(self):
        for f in [self.tempfile, self.conversion_filepath]:
            try:
                os.remove(f)
            except OSError:
                pass

    def download(self):
        logging.debug(msg=f"Downloading episode: {self}")
        response = requests.get(self.url, stream=True)

        with open(self.tempfile, 'wb') as handle:
            for data in response.iter_content(chunk_size=102400):
                handle.write(data)
        logging.debug(msg=f"Downloaded episode: {self}")

    def collect_audio_metadata(self) -> Tuple[str, int]:
        try:
            meta = pymediainfo.MediaInfo.parse(self.filepath)
            return meta.general_tracks[0].audio_codecs, meta.general_tracks[0].overall_bit_rate
        except Exception:  # General error with what we are parsing, lets re-do it
            return "None", 0

    def do_it(self):
        if self.podcast == "" or self.completed:
            logging.debug(msg=f"No need to process this podcast episode: {self}")
            return
        self._create_folder()
        self.download()
        self.convert()
        self.completed = True
        self.cleanup_temps()
        logging.info(f"Completed: {self}")


def load_config() -> Dict:
    with open("podcasts.yaml", "r") as conf:
        return yaml.safe_load(conf)


def store_config(config):
    with open("podcasts.yaml", "w+") as conf:
        yaml.dump(config, conf)

    global CONFIG
    CONFIG = Box(config)


PODCAST_LOCK = threading.Lock()


def make_root_path() -> str:
    fol = os.path.expanduser(CONFIG.root_path)
    if not os.path.exists(fol):
        os.mkdir(fol)
    return fol


parser = argparse.ArgumentParser(description='')
parser.add_argument("-i", "--import", dest="import_path", type=str, default=None, required=False)
parser.add_argument("-t", "--threads", dest="threads", type=int, default=1, required=False)

args = parser.parse_args()


def import_opml(path):
    tree = ElementTree.parse(path)
    root = tree.getroot()

    existing_config = load_config()
    podcasts = {pod['podcast']['name'] for pod in existing_config['podcasts']}

    for ele in root.findall("body/outline/outline"):
        podname = ele.get("text").replace(" ", "_")

        if podname in podcasts:
            continue
        else:
            logging.info(f"Importing: {podname}")
            feed_url = ele.get("xmlUrl")
            existing_config['podcasts'].append({'podcast': {"name": podname, "feed": feed_url}})

    store_config(existing_config)


def get_podcasts():
    try:
        pods: PodcastSet[Podcast] = Podcast.unpickle_it()

        Podcast.instances = pods
        for p in Podcast.instances:
            p.update_episodes()
        if len(pods) == len(CONFIG.podcasts):
            return pods
    except OSError:
        pods = PodcastSet()

    for podcast in CONFIG.podcasts:
        if podcast.podcast.name not in Podcast.instances:
            try:
                pods.add(Podcast(podcast.podcast.feed))
            except HTTPError:
                pass
            except Exception as e:
                logging.exception(e)
        else:
            # Just to make it clear we do nothing here
            continue

    return pods


def main():
    global CONFIG
    CONFIG = Box(load_config())

    global ROOT_PATH
    ROOT_PATH = make_root_path()

    if args.import_path:
        import_opml(args.import_path)

    while True:
        CONFIG = Box(load_config())

        podcasts = get_podcasts()

        podcasts = sorted(podcasts, key=lambda x: x.title, reverse=True)
        episodes = []
        for podcast in podcasts:
            episodes.extend(podcast.episodes)

        episodes.sort(key=lambda x: x.completed, reverse=True)

        with tqdm(total=len(episodes), desc="Episodes", dynamic_ncols=True) as pbar:
            with ThreadPoolExecutor(max_workers=args.threads) as ep_ex:
                futures = [ep_ex.submit(ep.do_it) for ep in episodes]
                for _ in as_completed(futures):
                    pbar.update(1)

        time.sleep(CONFIG.search_interval)


if __name__ == '__main__':
    atexit.register(cleanup)
    atexit.register(Podcast.pickle_it)
    main()
