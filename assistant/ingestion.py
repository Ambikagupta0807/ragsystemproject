"""
Indexing one uploaded document.

Nothing is indexed until somebody uploads it. There is no folder that gets
scanned, because in a real company documents arrive one at a time, from people,
with whatever file name they happened to have.

So a document is identified by what it says, not by what it is called:

    same text, any file name     already indexed, nothing to do, nothing to pay
    same policy and version,
    different text               an edit, so the old chunks are replaced
    new policy or new version    indexed as a new document

That is why the content hash is unique in the database and the file name is
not. policy.md, policy-final.md and policy(2).md with the same text are one
document.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from assistant import config, repository
from assistant.models import Chunk, SourceDocument

REQUIRED_FIELDS = ("title", "department", "visibility", "version", "status", "effective", "updated")


@dataclass
class UploadResult:
    """What happened to one uploaded file."""

    filename: str
    outcome: str  # added, updated, duplicate
    title: str = ""
    version: int = 0
    chunks: int = 0
    detail: str = ""

    def line(self) -> str:
        if self.outcome == "duplicate":
            return f"{self.filename}: already indexed as {self.detail}, skipped"
        return (
            f"{self.filename}: {self.outcome} as {self.title} v{self.version}, "
            f"{self.chunks} chunks"
        )


class UploadError(ValueError):
    """The file cannot be indexed, with a reason a person can act on."""


# parse the uplaoded doc, 
# parse() --> document metadata, actual doc content, content hash

def parse(text: str, filename: str) -> SourceDocument:
    """Read the header block and the body out of an uploaded markdown file.

    Line endings are normalised before anything else. A file saved on Windows
    carries \\r\\n and the same file from a Mac carries \\n, and without this
    line those two would hash differently and be indexed twice.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if not text.lstrip().startswith("---"):
        raise UploadError("the file must start with a header block between --- markers")

    _, header, body = text.split("---", 2)

    fields = {}
    for line in header.strip().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()

    missing = [field for field in REQUIRED_FIELDS if field not in fields]
    if missing:
        raise UploadError("the header is missing: " + ", ".join(missing))

# now we have doc content, metadat, file name... and we convert evcerything into a SourceDocument object
    return SourceDocument(
        source=filename,
        title=fields["title"],
        department=fields["department"],
        visibility=fields["visibility"],
        version=int(fields["version"]),
        status=fields["status"],
        effective_from=datetime.strptime(fields["effective"], "%Y-%m-%d").date(),
        updated_on=datetime.strptime(fields["updated"], "%Y-%m-%d").date(),
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        body=body.strip(),
    )



## split doc into chunks

# In this project, we use TWO levels of splitting.
#
# LEVEL 1:
# Split document based on Markdown headings.
#
# LEVEL 2:
# Split large sections based on chunk size.
#
def split(document: SourceDocument) -> list[Chunk]:
    """Split on headings first, then on size.

    Heading first means a section is never cut in half and the heading travels
    with the chunk, which is what lets an answer cite an exact section.
    """
    by_heading = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section")])
    by_size = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )

    chunks = []
    for section in by_heading.split_text(document.body):
        name = section.metadata.get("section", document.title)
        for piece in by_size.split_text(section.page_content):
            chunks.append(Chunk(section=name, content=piece))

    return chunks




#Storing into vector db

def store(connection, text: str, filename: str) -> UploadResult:
    """Index one uploaded file. The only way documents enter this system."""
    document = parse(text, filename)

    # Same text, under any name. Nothing to do, and nothing to pay for.
    existing = repository.find_by_hash(connection, document.content_hash)
    if existing:
        return UploadResult(
            filename=filename,
            outcome="duplicate",
            detail=f"{existing['title']} v{existing['version']} (from {existing['source']})",
        )

    # Same policy and version, different text. This is an edit, so the old
    # chunks go and the new ones take their place.
    replacing = repository.find_by_version(connection, document.title, document.version)
    if replacing:
        repository.delete_document(connection, replacing["id"])

    chunks = split(document)
    if not chunks:
        raise UploadError("the document has no content under its headings")

    vectors = OpenAIEmbeddings(model=config.EMBEDDING_MODEL).embed_documents(
        [chunk.content for chunk in chunks]
    )

    document_id = repository.insert_document(connection, document)
    repository.insert_chunks(
        connection,
        document_id,
        [chunk.section for chunk in chunks],
        [chunk.content for chunk in chunks],
        vectors,
    )
    connection.commit()

    return UploadResult(
        filename=filename,
        outcome="updated" if replacing else "added",
        title=document.title,
        version=document.version,
        chunks=len(chunks),
    )
