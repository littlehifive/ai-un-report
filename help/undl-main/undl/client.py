import os
import xml.etree.ElementTree as E
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymarc
import requests
from loguru import logger
from pymarc import Record

import undl.consts as consts


class UNDLClient:
    """
    Client class for the United Nations Digital Library API.
    """

    verbose: bool
    query_cache: Dict[str, Any]
    id_cache: Dict[str, Any]
    record_cache: Dict[str, Any]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.query_cache = {}
        self.id_cache = {}
        self.record_cache = {}

    def query(
        self,
        prompt: str,
        outputFormat: str = consts.DEFAULT_API_FORMAT,
        lang: str = "en",
        outputFile: Optional[str] = None,
        searchId: Optional[str] = None,
        apiKey: Optional[str] = os.getenv("UN_API"),
    ) -> Dict[str, Any] | str | None:
        """
        Function to query the official UNDL API.

        Parameters
        ----------
        `prompt` : `str`
            The prompt to search for.
        `outputFormat` : `str`, optional
            The format of the API call, by default `consts.DEFAULT_API_FORMAT`
        `lang` : `str`, optional
            Language, by default `"en"`
        `outputFile` : `Optional[str]`, optional
            Where to store the output file, by default `None`
        `searchId` : `Optional[str]`, optional
            Search ID, by default `None`
        `apiKey` : `Optional[str]`, optional
            API key in the case we use the new URL, by default
            `os.getenv("UN_API")`

        Returns
        -------
        `Dict[str, Any] | str | None`
            The query results
        """
        if prompt in self.query_cache:
            logger.success(f"Found prompt '{prompt}' in cache.")
            return self.query_cache[prompt]

        logger.success(f"Querying official UNDL API for prompt '{prompt}'")

        params = {
            "p": prompt,
            "format": outputFormat,
            "ln": lang,
        }

        if self.verbose:
            logger.info(f"Querying UNDL API with params: {params}")

        result = self._query(
            params=params,
            outputFormat="marcxml",
            outputFile=outputFile,
            searchId=searchId,
            apiKey=apiKey,
        )

        self.query_cache[prompt] = result

        return result

    def getAllRecordIds(
        self,
        prompt: str,
        outputFile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Returns only the record IDs corresponding to the search results instead of
        the whole records.

        Result is of the form:

        ```json
        {
            "total": 100,
            "hits": [
                "123456",
                "234567",
                ...
            ]
        }
        ```

        Parameters
        ----------
        `prompt` : `str`
            The prompt to search for.
        `outputFile` : `Optional[str]`, optional
            Where to store the results, by default `None`

        Returns
        -------
        `Dict[str, Any]`
            The query results
        """

        if prompt in self.id_cache:
            logger.success(f"Found prompt '{prompt}' in cache.")
            return self.id_cache[prompt]

        logger.success(
            f"Querying official UNDL API for record IDs of prompt '{prompt}'"
        )
        params = {
            "p": prompt,
            "ln": "en",
        }

        if self.verbose:
            logger.info(f"Querying UNDL API with params: {params}")

        result = self._query(
            params=params,
            outputFormat="json",
            outputFile=outputFile,
        )

        self.id_cache[prompt] = result

        return result

    def queryById(
        self,
        recordId: str,
        outputFile: Optional[str] = None,
    ) -> Dict[str, Any] | str | None:
        """
        Function to query the API for a unique ID.

        Parameters
        ----------
        `id` : `str`
            The id to query for.
        `outputFormat` : `str`, optional
            Output format _from the API_, by default consts.DEFAULT_FORMAT
        `outputFile` : `Optional[str]`, optional
            File to which the script will save the query results, by default None
        `oldURL` : `bool`, optional
            Flag to use the old URL instead of the official one, by default True
        `apiKey` : `Optional[str]`, optional
            API key in the case we use the new URL, by default `os.getenv("UN_API")`

        Returns
        -------
        `Dict[str, Any] | str | None`
            The query results
        """

        if recordId in self.record_cache:
            logger.success(f"Found record ID '{recordId}' in cache.")
            return self.record_cache[recordId]

        logger.info(f"Querying UNDL API for unique ID '{recordId}'")

        params = {
            "recid": recordId,
            "of": "xm",
            "c": "Resource Type",
        }

        if self.verbose:
            logger.info(f"Querying UNDL API with params: {params}")

        result = self._queryUnofficial(
            params=params,
            outputFormat="marcxml",
            outputFile=outputFile,
        )

        self.record_cache[recordId] = result

        return result

    """
    Helper functions
    """

    def _query(
        self,
        params: Dict[str, Any],
        outputFormat: str = consts.DEFAULT_API_FORMAT,
        outputFile: Optional[str] = None,
        apiKey: Optional[str] = os.getenv("UN_API"),
        searchId: Optional[str] = None,
    ) -> List[Dict[str, Any]] | str | None:
        """
        General query function.

        Parameters
        ----------
        `params` : `Dict[str, Any]`
            Query parameters.
        `outputFormat` : `Optional[str]`, optional
            Output format – by default MARCXML, by default consts.DEFAULT_FORMAT
        `outputFile` : `Optional[str]`, optional
            File to which the output will be saved, by default None
        `apiKey` : `Optional[str]`, optional
            The API key to use in the case of the new URL, by default None

        Returns
        -------
        `List[Dict[str, Any]] | str | None`
            The query results
        """

        if searchId:
            params["search_id"] = searchId

        r = requests.get(
            consts.API_BASE_URL,
            params=params,
            headers={
                "content-type": "application/xml",
                "Authorization": f"Token {apiKey}",
            },
        )

        logger.debug(f"URL: {r.url}")
        logger.debug(f"Params: {params}")

        match outputFormat:
            case "marcxml":
                # Skip the namespace
                responseXML = E.fromstring(
                    r.text.replace('xmlns="http://www.loc.gov/MARC21/slim"', "")
                )

                total = int(responseXML.find("total").text)
                searchId = responseXML.find("search_id").text

                logger.info(f"Found {total} results.")
                logger.debug(f"Search ID: {searchId}")
                logger.debug(f"URL: {r.url}")

                path = Path.home() / ".undl"
                os.makedirs(path, exist_ok=True)

                filePath = path / "tmp.xml"
                with open(filePath, "w") as f:
                    f.write(
                        E.tostring(responseXML.find("collection"), encoding="unicode")
                    )

                parsedResponse = self.parseMARCXML(
                    xml=str(filePath),
                    outputFile=outputFile,
                    total=total,
                    searchId=searchId,
                )
                logger.success(
                    f"Query successful. Saved {len(parsedResponse['records'])} result(s) to {outputFile}."
                )
            case "json":
                parsedResponse = r.json()

                if outputFile:
                    import json

                    with open(outputFile, "w") as f:
                        json.dump(parsedResponse, f, indent=4, ensure_ascii=False)

                logger.success(
                    f"Query successful. Saved {len(parsedResponse['hits'])} result(s) to {outputFile}."
                )
            case _:
                raise NotImplementedError("Only MARCXML is supported for now.")

        return parsedResponse

    def _queryUnofficial(
        self,
        params: Dict[str, Any],
        outputFormat: str = consts.DEFAULT_FORMAT,
        outputFile: Optional[str] = None,
    ) -> List[Dict[str, Any]] | str | None:
        """
        General query function.

        Parameters
        ----------
        `params` : `Dict[str, Any]`
            Query parameters.
        `oldURL` : `bool`, optional
            Flag to use the old URL or new one, by default True
        `outputFormat` : `Optional[str]`, optional
            Output format – by default MARCXML, by default consts.DEFAULT_FORMAT
        `outputFile` : `Optional[str]`, optional
            File to which the output will be saved, by default None

        Returns
        -------
        `List[Dict[str, Any]] | str | None`
            The query results
        """
        r = requests.get(
            consts.BASE_URL,
            params=params,
        )

        if self.verbose:
            logger.debug(f"URL: {r.url}")

        response = r.content.decode("utf-8")

        path = Path.home() / ".undl"
        os.makedirs(path, exist_ok=True)

        filePath = path / "tmp.xml"
        with open(filePath, "w") as f:
            f.write(response)

        match outputFormat:
            case "marcxml":
                parsedResponse = self.parseMARCXML(
                    xml=str(filePath),
                    outputFile=outputFile,
                )
            case _:
                raise NotImplementedError("Only MARCXML is supported for now.")

        logger.success(
            f"Query successful. Saved {len(parsedResponse)} results to {outputFile}."
        )

        return parsedResponse

    def parseMARCXML(
        self,
        xml: str,
        outputFile: Optional[str] = None,
        total: Optional[int] = None,
        searchId: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parse MARCXML to JSON.

        Parameters
        ----------
        `xml` : `str`
            MARCXML string to parse.
        `outputFile` : `Optional[str]`, optional
            File to which the output will be saved, by default `None`.
            In the `None` case, the output will be returned.
        `total` : `Optional[int]`, optional
            Total number of results, by default `None`.
        `searchId` : `Optional[str]`, optional
            Search ID, by default `None`.

        Returns
        -------
        `List[Dict[str, Any]]`
            The parsed MARCXML in a JSON-like format.
        """
        output = {
            "total": total,
            "search_id": searchId,
            "records": [],
        }
        records: List[Record] = pymarc.parse_xml_to_array(xml)

        for record in records:
            record = {
                "id": self.extractFromMARC(record, "001"),
                "title": self.extractFromMARC(record, "245"),
                "alt_title": self.extractFromMARC(record, "239"),
                "location": self.extractFromMARC(record, "260", "a"),
                "symbol": self._getSymbol(record),
                "publication_date": self.extractFromMARC(record, "269"),
                "summary": self.extractFromMARC(record, "520"),
                "authors": self.extractFromMARC(
                    record,
                    "710",
                    "a",
                    collection=True,
                ),
                "description": self.extractFromMARC(record, "300"),
                "downloads": self._getDownloads(record),
                "subjects": self._getSubjects(record),
                "agenda": self.extractFromMARC(record, "991"),
                "collections": self._getCollections(record),
                "related_documents": self.extractFromMARC(
                    record,
                    "993",
                    collection=True,
                ),
            }

            # Remove empty fields
            record = {k: v for k, v in record.items() if v}

            output["records"].append(record)

        if outputFile:
            import json

            with open(outputFile, "w") as f:
                json.dump(output, f, indent=4, ensure_ascii=False)

        return output

    def _getSubjects(self, record: Record) -> Dict[str, Any]:
        """
        Helper function to extract subject out of a MARCXML record.

        Parameters
        ----------
        `record` : `Record`
            MARCXML record from which to extract the subjects.

        Returns
        -------
        `Dict[str, Any]`
            The subjects
        """
        subjects: Dict[str, List[str]] = {
            "unbist": [],
            "unbisn": [],
            "misc": [],
        }

        raw_subjects = record.subjects()

        for raw_subject in raw_subjects:
            subject = raw_subject.subfields_as_dict()
            if subject["2"] == ["unbist"]:
                subjects["unbist"].extend(subject["a"])
            elif subject["2"] == ["unbisn"]:
                subjects["unbisn"].extend(subject["a"])
            else:
                subjects["misc"].extend(subject["a"])

        return subjects

    def _getDownloads(self, record: Record) -> Dict[str, Any]:
        """
        Helper function to extract downloads out of a MARCXML record.

        Parameters
        ----------
        `record` : `Record`
            MARCXML record from which to extract the downloads.

        Returns
        -------
        `Dict[str, Any]`
            The downloads
        """
        links: Dict[str, Any] = self.extractFromMARC(
            record,
            "856",
            "u",
            collection=True,
        )

        langs = self.extractFromMARC(
            record,
            "856",
            "y",
            collection=True,
        )

        return {lang: link for lang, link in zip(langs, links)}

    def _getCollections(self, record: Record) -> Dict[str, Any]:
        """
        Get collections from the MARCXML record.
        See https://research.un.org/en/digitallibrary/export

        Parameters
        ----------
        `record` : `Dict[str, Any]`
            MARCXML record from which to extract the collections.

        Returns
        -------
        `Dict[str, Any]`
            The collections
        """

        resource_type = record["989"].subfields_as_dict() if "989" in record else {}
        un_bodies = record["981"].subfields_as_dict() if "981" in record else {}

        collections = {
            "resource_type": [v[0] for v in resource_type.values()],
            "un_bodies": [v[0] for v in un_bodies.values()],
        }

        return collections

    def _getSymbol(self, record: Record) -> Optional[str | Dict[str, Any]]:
        """
        Get document symbol from the MARCXML record.
        See https://research.un.org/en/digitallibrary/export

        Parameters
        ----------
        `record` : `Dict[str, Any]`
            MARCXML record from which to extract the document symbol.

        Returns
        -------
        `str | None`
            The document symbol
        """

        symbol = self.extractFromMARC(record, "191", "a")

        if not symbol:
            symbol = self.extractFromMARC(record, "791", "a")

        return symbol

    def extractFromMARC(
        self,
        record: Record,
        field: str,
        subfield: Optional[str] = None,
        collection: bool = False,
    ) -> Optional[str | Dict[str, Any]]:
        """
        Extract a field from a MARC record.

        Parameters
        ----------
        `record` : `Dict[str, Any]`
            MARCXML record from which to extract the field.
        `field` : `str`
            MARC field to extract.
        `subfield` : `str`, optional
            MARC subfield to extract. None by default.

        Returns
        -------
        `Optional[str | Dict[str, Any]]`
            The extracted field.
        """
        try:
            result = list(
                map(
                    lambda x: x[subfield] if subfield else x.format_field(),
                    record.get_fields(field),
                )
            )

            if not collection and result:
                result = result[0]
            return result

        except Exception as e:
            if self.verbose:
                logger.error(f"Could not extract field {field} from MARC record. {e}")
            return None
