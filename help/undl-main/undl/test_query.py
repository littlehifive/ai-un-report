# -*- coding: utf-8 -*-
"""
UN Digital Library Tind Search API

Created on Wed May 10 18:23:43 2023

@author: Catherine
"""

import json

from loguru import logger

from undl.client import UNDLClient


def testQuery() -> None:
    """
    Test the query function.
    """
    client = UNDLClient(verbose=True)
    outputFile = "downloads/test.json"
    client.query(
        prompt="Women in peacekeeping",
        outputFile=outputFile,
    )

    with open(outputFile, "r") as f:
        data = json.load(f)

    assert len(data["records"]) == min(100, data["total"])


def testGetIds() -> None:
    """
    Test the getAllRecordIds function.
    """
    client = UNDLClient(verbose=True)
    outputFile = "downloads/test_record_ids.json"
    client.getAllRecordIds(
        prompt="Women in peacekeeping",
        outputFile=outputFile,
    )

    with open(outputFile, "r") as f:
        data = json.load(f)

    assert len(data["hits"]) == data["total"]


def testById() -> None:
    """
    Test the queryById function.
    """
    client = UNDLClient(verbose=True)
    outputFile = "downloads/test_by_id.json"
    client.queryById(
        recordId="515307",
        outputFile=outputFile,
    )

    with open(outputFile, "r") as f:
        data = json.load(f)

    logger.debug(json.dumps(data, indent=4))

    assert len(data["records"]) == 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", help="Prompt string to search for.")
    args = parser.parse_args()

    testQuery(args.prompt)
