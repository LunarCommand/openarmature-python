# Spec: pipeline-utilities §6.3 cause chain (proposal 0068) + §6.4 cause-chain
# classification (proposal 0074). The cause-chain types (CauseLink, the per-link
# record; CaughtException, the derived single category / message over the chain)
# and the public classification primitive that produces them live together here.
# §6.4 promotes the carrier-skipping cause-fidelity walk (§6.3) to a public,
# named primitive shared by §6.1 retry, §6.3 isolation, and consumers, so a
# carrier-wrapped failure classifies identically everywhere instead of each site
# re-deriving the walk subtly differently.

"""Cause-chain classification (types + public primitive).

A failure that crosses a subgraph / fan-out / branch boundary is wrapped by
the engine in one or more ``node_exception`` carriers. ``classify_cause_chain``
walks an exception's ``__cause__`` chain, records one :class:`CauseLink` per
exception (flagging those carriers), and derives the single failure category
the chain represents: the outermost non-carrier link's category, resolved
*through* the carriers. The result is a :class:`CaughtException` carrying the
ordered chain plus that derived ``category`` / ``message``.

This is the classification ``FailureIsolationMiddleware`` reports as
``caught_exception`` and the category vocabulary ``RetryMiddleware``'s
classifier matches against; exposing it publicly lets a ``catch`` set, a custom
``predicate``, a router, or a metric classify a carrier-wrapped failure the way
the framework does.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import NodeException


@dataclass(frozen=True)
class CauseLink:
    """One link in a caught exception's resolved cause chain.

    - ``category``: the link's failure category when it carries one (a
      string), else ``None``.
    - ``message``: the link's own message (the ``str`` of the exception).
    - ``carrier``: ``True`` when the link is an engine-applied
      ``node_exception`` carrier wrapper, ``False`` for an ordinary
      (non-carrier) exception.
    """

    category: str | None
    message: str
    carrier: bool


@dataclass(frozen=True)
class CaughtException:
    """A classified exception cause chain.

    The result of :func:`classify_cause_chain`, and the record
    ``FailureIsolatedEvent.caught_exception`` carries.

    - ``category``: the derived single failure category, the outermost
      non-carrier link whose category is a non-empty string, or ``None``
      when no non-carrier link carries one.
    - ``message``: the message of the link ``category`` is derived from,
      or (when no link carries a category) the outermost non-carrier
      link's message.
    - ``chain``: the ordered cause chain, outermost (the classified
      exception, index 0) to innermost (the originating raise), one
      :class:`CauseLink` per exception.
    """

    category: str | None
    message: str
    chain: tuple[CauseLink, ...]


def classify_cause_chain(exc: Exception) -> CaughtException:
    """Classify ``exc`` by walking its ``__cause__`` chain.

    Records one ``CauseLink`` per exception from ``exc`` (outermost) to the
    originating raise (innermost), flagging ``node_exception`` carriers, and
    derives the single category / message the chain represents (the outermost
    non-carrier categorized link).
    """
    chain = _build_cause_chain(exc)
    category, message = _derive_cause(chain)
    return CaughtException(category=category, message=message, chain=chain)


def _build_cause_chain(exc: Exception) -> tuple[CauseLink, ...]:
    # Walk the ``__cause__`` chain from the caught exception (outermost) to the
    # originating raise (innermost), one CauseLink per exception. A graph-engine
    # §4 ``node_exception`` carrier (NodeException and subtypes such as
    # ParallelBranchesBranchFailed) the engine applies at a non-node placement
    # (§9.7 instance / §11.7 branch / §9.6 / §11.6 parent-node middleware) is
    # flagged ``carrier=True``. Traverse only BaseException instances (a
    # non-exception ``__cause__`` ends the walk, per §6.3) and guard against a
    # cyclic chain so a malformed chain can't hang or crash the degrade path.
    links: list[CauseLink] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        category = getattr(current, "category", None)
        links.append(
            CauseLink(
                category=category if isinstance(category, str) and category else None,
                message=str(current),
                carrier=isinstance(current, NodeException),
            )
        )
        current = current.__cause__
    return tuple(links)


def _derive_cause(chain: tuple[CauseLink, ...]) -> tuple[str | None, str]:
    # Derived single ``category`` / ``message`` (§6.3, proposal 0068): the
    # OUTERMOST non-carrier link whose category is a non-empty string -- so a
    # deliberately re-categorized surface error wins, while an uncategorized
    # surface error resolves to the categorized cause beneath it (the same chain
    # §6.1's default classifier consults, so the reported category agrees with
    # what retry acted on). When no non-carrier link carries a category, the
    # category is null and the message is the outermost non-carrier link's. The
    # all-carrier fallback is defensive -- failure isolation always catches a
    # non-carrier or wraps one, so a chain with no non-carrier link should not
    # arise.
    surface: CauseLink | None = None
    for link in chain:
        if link.carrier:
            continue
        if surface is None:
            surface = link
        if isinstance(link.category, str) and link.category:
            return link.category, link.message
    if surface is not None:
        return None, surface.message
    return None, chain[0].message if chain else ""


__all__ = [
    "CauseLink",
    "CaughtException",
    "classify_cause_chain",
]
