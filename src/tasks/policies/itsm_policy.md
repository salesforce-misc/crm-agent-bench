# IT Service Management (ITSM) Domain — Operating Policy

1. Identify the ticket before reading or mutating it; never guess an identifier.
2. Check the requesting user's permission before destructive operations.
3. Check dependencies before deletion. A blocked or unauthorized deletion must
   leave tickets, dependencies, and session audit state unchanged.
4. Assignment, status, comments, escalation, and closure apply only to the
   requested ticket; unrelated tickets must remain untouched.
5. Respect the canonical business-hours response and report timestamps and
   ticket fields exactly as returned by tools.
6. Never claim a mutation succeeded when a canonical tool rejected it.
