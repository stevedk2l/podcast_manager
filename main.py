import argparse
import json
import logging
import os
import ssl
import tempfile
import time
import urllib
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from functools import lru_cache
from logging import config
from typing import List
from typing import Union
from urllib.error import HTTPError
from urllib.request import Request
from xml.etree import ElementTree

import certifi
import ffmpeg
import podcastparser
import pymediainfo as pymediainfo
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
            "format": "%(asctime)s  [%(levelname)s] %(module)s:%(lineno)d %(threadName)s)) -- %(message)s",
            "datefmt": "%d-%m-%Y %I:%M:%S"
        }
    },
}

config.dictConfig(log_config)

CONFIG: Box
ROOT_PATH: str = ""

format_rectifier = {"opus": "libopus"}


class Podcast:
    def __init__(self, pod):
        self.title: str = pod["title"].replace(" ", "_")
        self.episodes: List[PodcastEpisode] = self.create_episodes(pod["episodes"])
        logging.debug(f"Imported podcast: {self.title}")

    def create_episodes(self, parsed_feed) -> List['PodcastEpisode']:
        eps = []
        for ep in parsed_feed:
            eps.append(PodcastEpisode(self, ep))
        return eps

    def process_episodes(self):
        for ep in self.episodes:
            logging.debug(f"Processing: {self.title} -- {ep.name}")
            ep.do_it()

    def __str__(self):
        return self.title


UNITS = {"B": 1, "K": 1024, "M": 1048576}


def parse_size(size) -> int:
    unit = size[-1].upper()
    number = size[:-1]
    return int(float(number) * UNITS[unit])


@lru_cache(maxsize=1)
def get_req_info():
    return CONFIG.format, parse_size(CONFIG.bitrate)


class PodcastEpisode:
    def __init__(self, podcast, ep):
        self.podcast: Podcast = podcast
        self.name: str = ep['title']
        self.released: int = ep['published']
        self.url: str = ep['enclosures'][0]['url']
        self.tempfile: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(delete=False)
        self.guid: str = ep['guid']
        self._requested_format, self._requested_bitrate = get_req_info()

        self.filepath = os.path.join(ROOT_PATH,
                                     self.podcast.title,
                                     f"{self.podcast.title}__{self.released}__{self.name}.{self._requested_format}")

        self._completed = os.path.exists(self.filepath)

        if self._completed:
            self.format, self.bitrate = self.collect_audio_metadata()
            req_format, req_bitrate = get_req_info()
            if not ((self.format.lower() == req_format.lower()) or (
                    req_bitrate * 0.9 <= self.bitrate <= req_bitrate * 1.1)):
                logging.warning(f"Deleting: {self} as it doesn't match format and/or bitrate rules")
                self._completed = False
                logging.info(f"Marking: {self} as not complete")
                os.remove(self.filepath)
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
        logging.info(f"Creating folder: {folder}")
        if not os.path.exists(folder):
            os.mkdir(folder)

    def convert(self):
        logging.debug(msg=f"Converting episode: {self}")
        ffmpeg.input(self.tempfile.name).output(self.filepath,
                                                format=self._requested_format,
                                                acodec=format_rectifier[self._requested_format],
                                                ac=1,
                                                audio_bitrate=self._requested_bitrate) \
            .run(capture_stdout=True, capture_stderr=True)
        logging.debug(msg=f"Converted episode: {self}")
        os.remove(self.tempfile.name)
        self.format = self._requested_format
        self.bitrate = self._requested_bitrate
        self._requested_format, self._requested_bitrate = None, None

    def download(self):
        logging.debug(msg=f"Downloading episode: {self}")
        response = requests.get(self.url, stream=True)

        with self.tempfile as handle:
            for data in response.iter_content(chunk_size=102400):
                handle.write(data)
        logging.debug(msg=f"Downloaded episode: {self}")

    def collect_audio_metadata(self) -> (str, str):
        try:
            meta = pymediainfo.MediaInfo.parse(self.filepath)
            return meta.general_tracks[0].audio_codecs, meta.general_tracks[0].overall_bit_rate
        except Exception:  # General error with what we are parsing, lets re-do it
            return None, None

    def do_it(self):
        if self.podcast == "" or self._completed:
            logging.debug(msg=f"No need to process this podcast episode: {self}")
            return
        self._create_folder()
        self.download()
        self.convert()
        self._completed = True


def load_config():
    with open("podcasts.yaml", "r") as conf:
        return yaml.safe_load(conf)


def store_config(config):
    with open("podcasts.yaml", "w+") as conf:
        yaml.dump(config, conf)

    global CONFIG
    CONFIG = Box(config)


def create_podcast(feed):
    with urllib.request.urlopen(feed, context=ssl.create_default_context(cafile=certifi.where())) as response:
        pod = podcastparser.parse(feed, response)
        return Podcast(pod)


def make_root_path():
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


def main():
    global CONFIG
    CONFIG = Box(load_config())

    global ROOT_PATH
    ROOT_PATH = make_root_path()

    if args.import_path:
        import_opml(args.import_path)

    while True:
        CONFIG = Box(load_config())

        podcasts = set()
        for podcast in CONFIG.podcasts:
            try:
                podcasts.add(create_podcast(podcast.podcast.feed))
            except HTTPError:
                pass
            except Exception as e:
                logging.exception(e)

        episodes = []
        for podcast in podcasts:
            episodes.extend(podcast.episodes)

        with tqdm(total=len(episodes), desc="Episodes", dynamic_ncols=True) as pbar:
            with ThreadPoolExecutor(max_workers=args.threads) as ex:
                futures = [ex.submit(ep.do_it) for ep in episodes]
                for _ in as_completed(futures):
                    pbar.update(1)

        time.sleep(CONFIG.search_interval)


if __name__ == '__main__':
    main()
