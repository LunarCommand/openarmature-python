# Patterns

Recipes for things people keep asking the framework to do but
that compose cleanly from existing primitives.

The split between [Concepts](../concepts/index.md) and Patterns is
intentional: Concepts explain *what OpenArmature is* — typed state,
nodes, edges, middleware, checkpointing, observers. Patterns
explain *ways to use it* — opinionated shapes for common
downstream questions like "how do I run an agent loop?" or "how do
I skip work that's already been done?".

## When to read which

- You don't know what a `State` is, or how nodes and edges fit
  together → start with [Concepts](../concepts/index.md).
- You know the primitives but you're asking "how do I do X with
  them?" → look here.

Patterns are user-level recipes, not framework contracts. New
patterns can be added without spec coordination — they're how-to
docs composing existing primitives.

## The catalog

- [Parameterized entry point](parameterized-entry-point.md) —
  start the graph at an arbitrary node via state-driven routing.
- [Tool dispatch as node](tool-dispatch-as-node.md) — model an
  agent tool-call loop as a graph cycle.
- [Session as checkpoint resume](session-as-checkpoint-resume.md) —
  carry multi-turn agent state across turns using the existing
  checkpointer.
- [Bypass if output exists](bypass-if-output-exists.md) —
  short-circuit a node whose external output already exists, via
  middleware.
