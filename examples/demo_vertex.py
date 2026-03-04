"""Vertex AI extraction example.

Requires Google Cloud auth (``gcloud auth application-default login``)
or a service account file via ``GOOGLE_APPLICATION_CREDENTIALS``.

You can also pass credentials explicitly::

    result = extract(
        text,
        MovieReview,
        model="vertex:gemini-2.5-flash",
        provider_kwargs={
            "project_id": "my-project",
            "region": "us-central1",
            "service_account_file": "/path/to/sa.json",
        },
    )
"""

from pydantic import BaseModel

from parsantic import extract


class MovieReview(BaseModel):
    title: str
    rating: float
    sentiment: str


text = """
The new Dune movie was absolutely breathtaking. The cinematography and
Hans Zimmer's score created an immersive experience. I'd rate it 9.2
out of 10. Truly a masterpiece of modern sci-fi filmmaking.
"""

if __name__ == "__main__":
    result = extract(
        text,
        MovieReview,
        model="vertex:gemini-2.5-flash",
        provider_kwargs={"project_id": "my-gcp-project", "region": "us-central1"},
    )
    print(result.value)
