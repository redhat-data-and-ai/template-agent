"""File attachments: let a user give the agent a document.

Files land in the agent's own virtual filesystem under a configured
mount path, so the existing read/grep/glob tools reach them with no new
tool and no new permission surface.
"""
