import argparse
import json
from typing import Any, Dict

from loguru import logger

from undl.client import UNDLClient


def main(args: Dict[str, Any]) -> None:
    """
    Runs a query against the UN Digital Library API.

    Parameters
    ----------
    `args` : `Dict[str, Any]`
        Dictionary of CLI arguments passed to the script.
    """
    client = UNDLClient(verbose=args["verbose"])

    if args["id"]:
        client.queryById(
            recordId=args["id"],
            outputFormat=args["format"],
            outputFile=args["output"],
        )
    else:
        client.query(
            prompt=args["query"],
            outputFile=args["output"],
        )


def parse_args() -> Dict[str, Any]:
    """
    Parse CLI arguments.

    Returns
    -------
    `Dict[str, Any]`
        Dictionary of CLI arguments passed to the script.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Query the UN Digital Library API with a Python wrapper.",
    )
    parser.add_argument(
        "-q",
        "--query",
        help="Query string to search for.",
    )
    parser.add_argument(
        "--id",
        help="ID of the record to search for.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output.",
    )
    parser.add_argument(
        "--remove-empty",
        action="store_true",
        help="Removes empty field from output JSON.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file name.",
        default="./downloads/output.json",
    )

    parser.add_argument(
        "-f",
        "--format",
        type=str,
        help="Output format",
        default="marcxml",
    )

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    logger.debug(f"Args:\n{json.dumps(args, indent=2)}")
    main(args=args)
