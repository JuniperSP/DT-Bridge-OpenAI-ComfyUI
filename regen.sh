#!/bin/sh
# Regenerate the Python bindings in ./gen from ./proto and ./schema.
# Tools (flatc / protoc) are downloaded into ./tools by setup below if missing.
set -eu
cd "$(dirname "$0")"

if [ ! -x tools/flatc ] || [ ! -x tools/bin/protoc ]; then
  echo "missing tools/flatc or tools/bin/protoc - fetching them" >&2
  echo "(they are not committed; ./setup.sh --tools downloads and regenerates)" >&2
  if [ -x ./setup.sh ] && [ -z "${DT_REGEN_FROM_SETUP:-}" ]; then
    exec ./setup.sh --tools --yes
  fi
  exit 1
fi

printf 'attribute "primary";\nattribute "indexed";\n' > schema/config_attrs.fbs
cat schema/config.fbs >> schema/config_attrs.fbs

printf 'attribute "primary";\nattribute "indexed";\n' > schema/th_attrs.fbs
cat schema/tensor_history.fbs >> schema/th_attrs.fbs

./tools/bin/protoc -I proto --python_out=gen proto/imageService.proto
./tools/flatc --python -o gen schema/config_attrs.fbs
./tools/flatc --python -o gen schema/th_attrs.fbs

echo "regenerated $(ls gen | wc -l | tr -d ' ') files in gen/"
