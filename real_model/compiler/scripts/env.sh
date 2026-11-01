# Source this from the repository root.
export PATH="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin:}$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/toolchain/usr/lib/llvm-19/lib:$PWD/toolchain/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
