# Third-party dependencies

## RoboCasa derived task manifest

`alphaapollo/data_preprocess/robotics/snapshots/robocasa365_tasks.jsonl`
contains task instructions and horizon fields derived from RoboCasa's public
code at `921c9a5`. Those upstream portions retain RoboCasa's MIT attribution;
the complete pinned [license](robocasa/LICENSE) is included. The adjacent
metadata and [provenance guide](../alphaapollo/data_preprocess/robotics/UPSTREAM.md)
record generation inputs and boundaries. RoboCasa demonstrations, scene assets,
MuJoCo and robosuite source are not included in this snapshot.

## Optional Learning dependency

The shared Learning runtime keeps a pinned `verl/` submodule. Chips tests and
simulator workflows do not require it. The former WebShop site belonged to an
unrelated environment example and is not part of this branch.

## Source distribution

Circuit Harness derives from AlphaApollo v3, with the initial Chips baseline
`264a00cba2048bbf06153ea32f4b0271d07cdaeb` in
[AlphaApollo-v3-dev](https://github.com/AndrewZhou924/AlphaApollo-v3-dev).
Keep the root [Apache-2.0 license](../LICENSE) and [attribution notice](../Notice.txt)
when distributing a source snapshot. The repository name change does not change
the `alphaapollo` package identity or remove upstream attribution.

`git archive` does not include the `verl/` submodule contents. A source candidate's
manifest must record its URL from `.gitmodules` and exact Git tree entry before
export. To use Learning from an archive, obtain that revision separately from
the declared upstream and place its checkout at `third_party/verl`; keep its own
license and notices. Do not substitute the current upstream default branch.

Robotics preparers and their derived task manifest record their source and data
boundaries in [UPSTREAM.md](../alphaapollo/data_preprocess/robotics/UPSTREAM.md).
EVAS, circuit benchmark bundles, PDKs and commercial EDA installations are
external inputs, not vendored dependencies. Review any proposed redistribution
of those materials separately from this source package.

## Adapted source

- `pi/` contains the MIT license for `earendil-works/pi`. AlphaApollo's
  `common/execution/tools/truncate.py` and
  `common/execution/tools/output_accumulator.py` adapt Pi's coding-agent
  tail-truncation and streaming accumulation behavior to Python; no Pi runtime
  dependency is vendored.
