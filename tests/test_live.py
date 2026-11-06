"""The whole flow against the hosted API, on a grid made for it and deleted
after. Runs only with ``HASHGRID_LIVE=1``: it creates a real grid."""

import os
import time

import pytest

from hashgrid import DNA, NotFound, TooLarge

pytestmark = pytest.mark.skipif(os.environ.get("HASHGRID_LIVE") != "1", reason="set HASHGRID_LIVE=1")

CITIES = ["Lisbon", "Barcelona", "Naples", "Split", "Valletta", "Nice", "Dubrovnik", "Porto",
          "Prague", "Vienna", "Budapest", "Krakow", "Munich", "Zurich", "Madrid", "Berlin"]
COASTAL = set(CITIES[:8])


def test_live_flow():
    dna = DNA(DNA.signup("sdk-live-test"))
    try:
        assert dna.info().embedder
        dna.put_many({f"cities/{c}": c for c in CITIES})
        assert dna.put("records/1", {"name": "Lisbon"}).type == "application/json"
        assert dna.get("records/1") == {"name": "Lisbon"}
        assert len(list(dna.items("cities/"))) == len(CITIES)
        with pytest.raises(TooLarge):
            dna.put("big", "x" * 9000)

        query = dna.query("a city for a weekend away")
        for _ in range(6):
            results = query.results(limit=3, prefix="cities/")
            for r in results:
                query.grade(r, 1.0 if r.key.split("/")[1] in COASTAL else -1.0)
            time.sleep(3)  # DNA trains in the background
        last = query.results(limit=3, prefix="cities/")
        assert sum(r.key.split("/")[1] in COASTAL for r in last) >= 2

        dna.delete("records/1")
        with pytest.raises(NotFound):
            dna.get("records/1")
    finally:
        dna.delete_grid()
