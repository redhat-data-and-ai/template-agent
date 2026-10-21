# Document extraction

Uploaded files and indexed sources are turned into text by one of three
parsers. The default needs nothing installed and nothing deployed.

| Engine | What it is | Cost |
|---|---|---|
| `builtin` *(default)* | pypdf, python-docx, openpyxl, python-pptx — already in the image | none |
| `docling` | layout-aware, good on scanned PDFs, tables and multi-column papers | a service you run |
| `tika` | broad format coverage, mature, cheap to run | a service you run |

**This template never hosts the extractor.** There is no container in
any overlay and no new Python dependency — `engines.py` is an HTTP
client and nothing else. Docling's image is ~2 GB with a model cache
that has to stay warm; that is a real operational cost, and most
deployments do not get it back in parse quality. Run one if you already
have one.

## Turning it on

```yaml
# config/agent/runtime/agent.yaml
extraction:
  engine: docling
  url: http://docling:5001
```

Locally:

```sh
podman run -p 5001:5001 quay.io/docling-project/docling-serve
```

Then check it before you need it:

```sh
curl -H "Authorization: Bearer $TOKEN" localhost:5002/attachments/extraction
{"engine":"docling","reachable":true,"detail":"docling reachable", ...}
```

## What the implementation guarantees

**An extractor never fails an upload.** It is a quality improvement,
not a dependency. Every failure path — unreachable, timeout, HTTP
error, conversion failure, empty output — degrades to the built-in
parser. Set `fallback: false` to make it a hard dependency; that is a
decision a deployment is allowed to make and should not make by
accident.

**A degraded parse is visible.** The upload response carries `engine`
and `degraded` per file, the attachment chip in the UI reads
*(basic parse)*, and the model sees it in the provenance line. This is
the one bug we actually hit the first time round: a pod that had the
config but not the client, falling back silently, indistinguishable
from working.

**Better output, not just different output.** When the remote result is
under 200 characters the built-in parser runs too, and whichever read
more text wins. A layout model returning `<no text content>` for a PDF
pypdf handles fine is a regression, and only a comparison catches it.
The local parse is skipped when the remote result is substantial, so
this costs nothing in the normal case.

**Only formats that benefit are sent.** `csv`, `json`, `md`, `txt` and
`xlsx` stay local — they are parsed exactly right here, and a network
round trip cannot improve on exactly right. Files over `max_bytes`
(20 MB) stay local too: shipping 50 MB to a shared extractor to read
its first page is how one upload starves everyone else's.

**Failures inside a 200 are caught.** Docling reports
`status: failure` or `skipped` in the body of a successful HTTP
response. Trusting the status code stores the error message as the
document — and it is then indexed, retrieved, and answered from.

**Everything is bounded.** `timeout_seconds` defaults to 120. An
extractor that has wedged is slower than one that is merely slow, and
an upload that never returns is worse than one that degrades.

## Where it matters most

Ingestion, not chat. An indexed document is parsed once and read on
every later search, so a bad parse is wrong forever and a good one is
paid for once. Both `POST /threads/{id}/attachments` and
`POST /knowledge/documents` use the same path.
