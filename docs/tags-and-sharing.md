# Tags and sharing

Two ways to organise conversations and one way to publish them.
Projects already give a conversation a home; these cover the rest.

## Tags

| Endpoint | |
|---|---|
| `PUT /threads/{id}/tags` | Replace a conversation's tags |
| `GET /threads/{id}/tags` | Read them |
| `GET /tags` | Every tag you have used, with counts |
| `GET /tags/{tag}/threads` | Conversations carrying a tag |

Tags live in `thread.metadata_json`, not in a new table. The column
already has a GIN index on `jsonb_path_ops`, which is exactly the index
a containment query over a tag array wants — so filtering is an index
scan with no migration, and deleting a thread deletes its tags for
free.

Rules worth knowing: tags are folded to lower case (`Billing` and
`billing` are one tag), whitespace is collapsed, the list comes back
sorted, max 32 characters and 20 tags per conversation. Punctuation is
rejected.

`PUT` replaces rather than adds, because the UI edits a set and an
add/remove pair would need the client to diff against state it may not
have. Ownership is checked inside the `UPDATE`, not with a `SELECT`
first — two statements leave a window where the thread changes hands
in between. A thread you do not own returns 404, not 403; a 403 would
confirm it exists.

## Sharing

| Endpoint | |
|---|---|
| `POST /threads/{id}/share` | Publish a snapshot, returns a link |
| `GET /threads/{id}/share` | Is it shared, and has the link been opened |
| `DELETE /threads/{id}/share` | Revoke |
| `GET /shares` | Everything you have published |
| `GET /shared/{share_id}` | Read a shared conversation |

Four defaults, each chosen against the obvious alternative:

**A snapshot, not a live view.** A link that keeps updating as you keep
talking publishes messages you had not written when you shared it.
Nobody reasons about a link that way. Re-sharing takes a new snapshot,
explicitly, and invalidates the old link.

**Authenticated by default.** `scope: "authenticated"` means any
signed-in user with the link can read it. `scope: "public"` exists and
is never the default — "anyone with the link" on an internal tool
carrying customer data is a decision somebody should have to make.

**Links expire**, 30 days by default, 365 maximum. open-webui's shares
do not expire at all; the risk is not the moment of sharing, it is the
year afterwards.

**Attachments are named, not embedded.** A snapshot that carried
uploaded files would re-publish documents through a URL their owner
never saw. The reader sees that `contract.pdf` was part of the
conversation; they do not get `contract.pdf`.

### What a snapshot contains

An allowlist, because thread state grows new keys as middleware is
added and a denylist would publish each new one by default.

| Kept | Dropped |
|---|---|
| Human and AI message text | Tool messages — they carry retrieved documents and file contents |
| Attachment *names* | System messages — prompt internals |
| Title, capture time | Image and audio blocks — attachment bytes under another name |
| | Memories, todos, and every other state channel |

Messages are capped at 500 per snapshot and 20,000 characters each.

### Audit

`GET /threads/{id}/share` reports `views` and `last_viewed_at`. A share
nobody opened is a share worth revoking, and without the number there
is no way to tell one from a link that circulated.

Expired and revoked are reported apart — HTTP 410 either way, but with
different text, because "ask them to share it again" and "they took it
down" are different things for the reader to do.
