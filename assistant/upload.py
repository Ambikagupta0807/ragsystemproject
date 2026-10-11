"""
PROJECT: Secure Enterprise RAG Assistant

FILE: upload.py

PURPOSE:
Upload company documents into our RAG database.

This file is the starting point for document ingestion
when we upload documents using the terminal.

Example commands:

    uv run upload.py sample-documents/refund-policy.md

    uv run upload.py sample-documents/*.md

IMPORTANT:

This file does not perform chunking or embedding itself.

Instead, it calls ingestion.store(), which we created
in the previous file.

That function handles:

    Document parsing
        -> Duplicate checking
        -> Chunking
        -> Embedding generation
        -> PostgreSQL storage

We can also upload documents using our web application later.
Both methods use the same ingestion service.
"""


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

# sys allows us to read command-line arguments.
#
# Example:
#
# uv run upload.py refund-policy.md
#
# sys.argv[0] = "upload.py"
# sys.argv[1] = "refund-policy.md"

import sys


# Path helps us work with file paths.
#
# Example:
#
# Path("sample-documents/refund-policy.md")
#
# We can use it to read the file content.

from pathlib import Path


# ingestion:
# Contains our document ingestion logic.
#
# repository:
# Contains database connection and SQL operations.

from assistant import ingestion, repository


# ============================================================
# 2. CONFIGURE TERMINAL ENCODING
# ============================================================

# Set terminal output encoding to UTF-8.
#
# WHY?
#
# Company documents may contain:
#
# - Special symbols
# - Currency symbols
# - Unicode characters
#
# UTF-8 helps display them correctly.
#
# This is particularly useful on Windows.

sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# 3. READ DOCUMENT PATHS FROM TERMINAL
# ============================================================

# sys.argv contains the command-line arguments.
#
# Example:
#
# uv run upload.py sample-documents/refund-policy.md
#
# sys.argv:
#
# [
#     "upload.py",
#     "sample-documents/refund-policy.md"
# ]
#
# sys.argv[1:] removes the script name.
#
# Path(argument) converts each filename into a Path object.
#
# The result is a list because users can upload
# multiple documents in a single command.

paths = [
    Path(argument)
    for argument in sys.argv[1:]
]


# ============================================================
# 4. CHECK WHETHER FILES WERE PROVIDED
# ============================================================

# If the user runs:
#
# uv run upload.py
#
# No document path is provided.
#
# In that case:
#
# 1. Print the instructions from the module docstring.
# 2. Stop the program.

if not paths:

    # __doc__ contains the documentation string
    # written at the beginning of this file.

    print(__doc__)

    # Exit the program with status code 1.
    #
    # 1 means the command could not complete successfully.

    raise SystemExit(1)


# ============================================================
# 5. CONNECT TO POSTGRESQL
# ============================================================

# repository.connection() provides a PostgreSQL connection.
#
# We created this function in repository.py.
#
# It uses a connection pool.
#
# WHY CONNECTION POOL?
#
# Instead of opening a completely new database connection
# every time, we reuse available connections.
#
# The "with" statement also ensures that the borrowed
# connection is returned to the pool afterwards.

with repository.connection() as connection:

    # --------------------------------------------------------
    # STEP 1: CREATE DATABASE TABLES
    # --------------------------------------------------------

    # Execute schema.sql through repository.py.
    #
    # This prepares the required database tables.
    #
    # Our project uses:
    #
    # users
    # documents
    # chunks
    # audit_events
    #
    # The schema also creates the required indexes.

    repository.create_schema(connection)


    # --------------------------------------------------------
    # STEP 2: PROCESS EACH DOCUMENT
    # --------------------------------------------------------

    # Loop through every file provided in the terminal.
    #
    # Example:
    #
    # refund-policy.md
    # shipping-policy.md
    # warranty-policy.md

    for path in paths:

        try:

            # ------------------------------------------------
            # STEP 3: READ AND UPLOAD DOCUMENT
            # ------------------------------------------------

            # path.read_text():
            # Reads the complete document content.
            #
            # encoding="utf-8":
            # Reads the document using UTF-8.
            #
            # path.name:
            # Returns only the filename.
            #
            # Example:
            #
            # path = sample-documents/refund-policy.md
            #
            # path.name = refund-policy.md
            #
            # ingestion.store():
            # Calls the ingestion service created previously.
            #
            # It handles parsing, chunking, embeddings
            # and storing the document in PostgreSQL.

            result = ingestion.store(

                connection,

                path.read_text(encoding="utf-8"),

                path.name
            )


            # ------------------------------------------------
            # STEP 4: DISPLAY UPLOAD RESULT
            # ------------------------------------------------

            # result is an UploadResult object.
            #
            # result.line() converts the result
            # into a readable message.
            #
            # Example:
            #
            # refund-policy.md: added as Refund policy v1,
            # 4 chunks

            print(result.line())


        # ----------------------------------------------------
        # STEP 5: HANDLE INVALID DOCUMENTS
        # ----------------------------------------------------

        # ingestion.UploadError is our custom exception.
        #
        # It is raised when the document is invalid.
        #
        # Example:
        #
        # Missing metadata
        # Missing header
        # No content under headings
        #
        # We display the error and continue with
        # the remaining documents.

        except ingestion.UploadError as error:

            print(
                f"{path.name}: rejected, {error}"
            )


    # --------------------------------------------------------
    # STEP 6: GET DATABASE STATISTICS
    # --------------------------------------------------------

    # After processing all documents,
    # ask PostgreSQL for the current counts.
    #
    # repository.counts() returns information such as:
    #
    # {
    #     "documents": 11,
    #     "active_documents": 10,
    #     "chunks": 44
    # }

    counts = repository.counts(connection)


# ============================================================
# 6. DISPLAY FINAL SUMMARY
# ============================================================

# Print an empty line for readability.

print()


# Display the total number of documents and chunks.
#
# Example:
#
# indexed: 11 documents (10 active), 44 chunks
#
# documents:
# Total number of indexed documents.
#
# active_documents:
# Number of documents marked active.
#
# chunks:
# Total number of stored text chunks.

print(
    f"indexed: {counts['documents']} documents "
    f"({counts['active_documents']} active), "
    f"{counts['chunks']} chunks"
)