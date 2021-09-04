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


class CustomSet(set):
    def __getitem__(self, item):
        for ele in self:
            if ele == item:
                return ele

class Podcast:
    instances = CustomSet()

    def __init__(self, pod):
        self.title: str = pod["title"].replace(" ", "_")
        if self in Podcast.instances:
            self.episodes = Podcast.instances[self].episodes
            self.update_episodes()
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
        ...

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
            return pickle.load(f)

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
        self.url: str = ep['enclosures'][0]['url']
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


def create_podcast(feed) -> Podcast:
    with urllib.request.urlopen(feed, context=ssl.create_default_context(cafile=certifi.where())) as response:
        pod = podcastparser.parse(feed, response)
        return Podcast(pod)


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


def generic_threaded_worker(iterable, function, args, description):
    with tqdm(total=len(iterable), desc=description, dynamic_ncols=True) as pbar:
        with ThreadPoolExecutor(max_workers=args.threads) as ep_ex:
            futures = [ep_ex.submit(getattr(x, function), args) for x in iterable]
            for _ in as_completed(futures):
                pbar.update(1)


def get_podcasts():
    try:
        pods = Podcast.unpickle_it()
        #if pods:
        #   return pods
    except OSError:
        pass

    podcasts = set()
    for podcast in CONFIG.podcasts:
        try:
            podcasts.add(create_podcast(podcast.podcast.feed))
        except HTTPError:
            pass
        except Exception as e:
            logging.exception(e)

    return podcasts


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
