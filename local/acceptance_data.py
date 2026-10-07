"""Download checksum-pinned public UCI data for opt-in product acceptance checks."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.request import urlopen
from zipfile import ZipFile

sources = {
    "bank": {
        "url": "https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip",
        "sha256": "e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4",
        "citation": "Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306",
        "changes": "Use bank.csv, the supplied 10% sample. Omit duration, which is unavailable before a call. Represent unknown categories as empty CSV cells. No client IDs are available to rule out repeated clients across splits.",
    },
    "bike": {
        "url": "https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip",
        "sha256": "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401",
        "citation": "Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894",
        "changes": "Use day.csv. Omit instant and the target components casual and registered. Split by dteday. Weather is observed, so this checks conditional demand estimation rather than a future weather forecast.",
    },
}


def prepare_dataset(name, destination):
    record = {**sources[name], "license": "CC BY 4.0", "licenseUrl": "https://creativecommons.org/licenses/by/4.0/"}
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "source.zip"
    if not archive.exists():
        with urlopen(record["url"], timeout=45) as response:
            content = response.read(5_000_001)
        if len(content) > 5_000_000:
            raise ValueError("Dataset archive exceeded 5 MB.")
    else:
        content = archive.read_bytes()
    if hashlib.sha256(content).hexdigest() != record["sha256"]:
        raise ValueError("UCI archive checksum changed. Review the source before updating this fixture.")
    archive.write_bytes(content)
    with ZipFile(io.BytesIO(content)) as outer:
        if name == "bank":
            with ZipFile(io.BytesIO(outer.read("bank.zip"))) as inner:
                raw = inner.read("bank.csv").decode("utf-8")
        else:
            raw = outer.read("day.csv").decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(raw), delimiter=";" if name == "bank" else ","))
    excluded = {"duration"} if name == "bank" else {"instant", "casual", "registered"}
    fields = [key for key in rows[0] if key not in excluded]
    dataset = destination / "train.csv"
    with dataset.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: "" if name == "bank" and row[key] == "unknown" else row[key] for key in fields} for row in rows)
    record.update(rows=len(rows), columns=fields, preparedSha256=hashlib.sha256(dataset.read_bytes()).hexdigest())
    (destination / "provenance.json").write_text(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    name, destination = sys.argv[1:]
    if name not in sources:
        sys.exit("Choose bank or bike.")
    print(json.dumps(prepare_dataset(name, Path(destination))))
