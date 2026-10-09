#!/usr/bin/env python3
"""cu2metal: the CUDA ports (cuda/*/src) as a Metal program, translated at build time.

    python3 metal/tools/cu2metal.py --root=cuda/af3/src/af3.cu --out=metal/build/af3

The CUDA sources stay the one source of truth for every model's conventions; this turns them into
  - HOST code (<out>/gen/<path under cuda/>): the same files, compiled as Objective-C++ against metal/runtime's
    cuda_runtime.h / cublas_v2.h / cuda_fp16.h, with every __global__ kernel replaced by a host stub that packs
    its arguments and dispatches its Metal twin, every `k<<<grid, block, smem, stream>>>(args)` rewritten to
    `(lf::setLaunch(grid, block, smem, stream), k(args))`, and every device-only function removed;
  - DEVICE code (<out>/kernels.metal): every kernel and device function as Metal Shading Language, through
    metal/runtime/prelude.metal's CUDA vocabulary (threadIdx, __syncthreads, __shfl_xor_sync, atomicAdd, half,
    bfloat16...). Every kernel becomes a TEMPLATE (a dummy parameter where CUDA had none), so nothing compiles
    until a launch asks for it: the runtime appends one explicit instantiation per specialisation used.
  - a kernel table (<out>/gen/lf_kernels.inc): the stubs' index into it, each kernel's Metal name and struct.

What it does not translate it says so: inline PTX (mma.sync, ldmatrix, cp.async), lambdas it cannot lower, and
anything named in metal/<port>/overrides.txt, which a hand-written Metal kernel or host function replaces.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
CUDA = os.path.join(REPO, "cuda")


# ---------------------------------------------------------------- lexing
def mask(src):
    """The source with comments as spaces and string/char literal contents as 'x' - same length, same newlines -
    so brace and paren matching can run on it and every span maps straight back onto the original."""
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if src[k] != "\n":
                    out[k] = " "
            i = j
        elif c == '"' or c == "'":
            # (a raw string R"(...)" - none in device code; host code keeps them verbatim)
            if c == '"' and i > 0 and src[i - 1] == "R":
                d = src.find("(", i)
                delim = ")" + src[i + 1:d] + '"'
                j = src.find(delim, d)
                j = n if j < 0 else j + len(delim)
                for k in range(i + 1, j - 1):
                    if src[k] != "\n":
                        out[k] = "x"
                i = j
                continue
            j = i + 1
            while j < n and src[j] != c:
                if src[j] == "\\":
                    j += 1
                j += 1
            for k in range(i + 1, min(j, n)):
                if src[k] != "\n":
                    out[k] = "x"
            i = j + 1
        else:
            i += 1
    return "".join(out)


def strip_comments(src):
    """The source with its comments removed (strings kept): a body joined onto one line as a macro must not have a
    `//` comment swallow the rest of it."""
    m = mask(src)
    out = []
    for c, mc in zip(src, m):
        # (the mask turns a comment's characters to spaces and a string's to 'x': a non-space masked to a space was a
        # comment)
        out.append(" " if mc == " " and not c.isspace() else c)
    return "".join(out)


def live_names(text):
    """The variables declared in `text` that are still in scope at its end: a stack of brace scopes, each holding what
    it declared (every declarator of a declaration, `float m[H], l[H], acc[H][D];` included)."""
    m = mask(text)
    scopes = [set()]
    stmt_start = 0
    i = 0
    def declare(stmt):
        st = stmt.strip()
        if not st or st.startswith("#"):
            return
        st = re.sub(r"^(for|while|if|switch)\s*\(", "", st)
        dm = re.match(r"(?:(?:const|constexpr|static|thread|device|threadgroup|unsigned|signed|volatile)\s+)*"
                      r"([A-Za-z_][\w:]*(?:\s*<[^;{}()]*>)?)\s*([*&]\s*)*(.*)$", st, re.S)
        if not dm or dm.group(1) in KEYWORDS - {"auto", "int", "float", "uint", "bool", "char", "short", "long", "ulong",
                                                  "ushort", "uchar", "half", "half2", "float2", "float3", "float4", "int2",
                                                  "int3", "int4", "uint2", "uint4", "unsigned", "signed"}:
            return
        rest = dm.group(3)
        if not rest or rest[0] in "(=+-[.<>!" or "=" == rest.strip()[:1]:
            return
        for decl in split_commas(rest):
            nm = re.match(r"\s*[*&]*\s*([A-Za-z_]\w*)\s*(\[|=|$|\{|\()", decl)
            if nm and nm.group(1) not in KEYWORDS:
                scopes[-1].add(nm.group(1))
    while i < len(m):
        c = m[i]
        if c == "{":
            declare(m[stmt_start:i])
            scopes.append(set()); stmt_start = i + 1
        elif c == "}":
            if len(scopes) > 1:
                scopes.pop()
            stmt_start = i + 1
        elif c == ";":
            declare(m[stmt_start:i]); stmt_start = i + 1
        elif c == "(":
            # a for-statement's init declares into the scope the loop body opens: count it into the current one
            close = match(m, i, "(", ")")
            head = m[stmt_start:i]
            if re.search(r"\b(for|if|while|switch)\s*$", head):
                # (a loop's or condition's own declarations end with its statement: never a lambda's capture)
                i = close + 1
                stmt_start = i
                continue
            i = close
        i += 1
    names = set()
    for sc in scopes:
        names |= sc
    return names


def match(m, i, open_c, close_c):
    """Index of the bracket closing the one at m[i]."""
    depth = 0
    for k in range(i, len(m)):
        if m[k] == open_c:
            depth += 1
        elif m[k] == close_c:
            depth -= 1
            if depth == 0:
                return k
    raise ValueError(f"unbalanced {open_c} at {i}")


def match_angle(m, i):
    """Index of the '>' closing the template '<' at m[i] (parentheses nest inside)."""
    depth = 0
    k = i
    while k < len(m):
        c = m[k]
        if c == "(":
            k = match(m, k, "(", ")")
        elif c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
            if depth == 0:
                return k
        k += 1
    raise ValueError("unbalanced <")


def split_commas(text):
    """Top-level comma split (parens, brackets, braces and template angles nest)."""
    parts, depth, cur = [], 0, ""
    angle = 0
    for i, c in enumerate(text):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "<" and depth == 0:
            # a template angle only after an identifier, not a comparison: heuristically, no space before it
            if i > 0 and (text[i - 1].isalnum() or text[i - 1] == "_"):
                angle += 1
        elif c == ">" and depth == 0 and angle > 0:
            angle -= 1
        if c == "," and depth == 0 and angle == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += c
    if cur.strip():
        parts.append(cur)
    return parts


# ---------------------------------------------------------------- top-level items
class Item:
    def __init__(self, start, end, kind, header=""):
        self.start, self.end, self.kind, self.header = start, end, kind, header

    def __repr__(self):
        return f"Item({self.kind}, {self.start}-{self.end})"


def items_of(m, start, end):
    """Top-level declarations of m[start:end]: preprocessor lines, namespace blocks (recursed into), and
    everything else up to a ';' or a definition's closing brace."""
    out = []
    i = start
    while i < end:
        while i < end and m[i].isspace():
            i += 1
        if i >= end:
            break
        if m[i] == "#":
            j = i
            while True:
                nl = m.find("\n", j)
                if nl < 0 or nl >= end:
                    nl = end
                    break
                if m[nl - 1] == "\\":
                    j = nl + 1
                    continue
                break
            out.append(Item(i, nl, "pp"))
            i = nl
            continue
        j = i
        depth = 0
        while j < end:
            c = m[j]
            if c == "(":
                j = match(m, j, "(", ")")
            elif c == "[":
                j = match(m, j, "[", "]")
            elif c == ";":
                out.append(Item(i, j + 1, "decl", m[i:j]))
                j += 1
                break
            elif c == "{":
                header = m[i:j]
                if re.match(r"\s*(inline\s+)?namespace\b[^{]*$", header) or re.match(r'\s*extern\s+"x"\s*$', header):
                    close = match(m, j, "{", "}")
                    out.append(Item(i, j + 1, "ns_open", header))
                    out.extend(items_of(m, j + 1, close))
                    out.append(Item(close, close + 1, "ns_close"))
                    j = close + 1
                    break
                close = match(m, j, "{", "}")
                k = close + 1
                is_function = re.search(r"\)\s*(const\s*)?(noexcept\s*)?(->\s*[\w:<>,\s*&]+)?\s*$", header) is not None \
                    and not re.search(r"=\s*$", header) and not re.search(r"\b(struct|class|union|enum)\b[^(]*$", header)
                if not is_function:
                    # a struct / enum / initialised variable: up to its ';'
                    while k < end and m[k] != ";":
                        if m[k] == "{":
                            k = match(m, k, "{", "}")
                        k += 1
                    k += 1
                out.append(Item(i, min(k, end), "def", header))
                j = min(k, end)
                break
            else:
                j += 1
        else:
            if m[i:end].strip():
                out.append(Item(i, end, "decl", m[i:end]))
            j = end
        i = j
    return out


def classify(it, m):
    if it.kind in ("pp", "ns_open", "ns_close"):
        return it.kind
    h = it.header if it.kind == "def" else m[it.start:it.end]
    fn = it.kind == "def" and "(" in h and not re.search(r"\b(struct|class|union|enum)\b[^(]*$", h)
    if re.search(r"\b__global__\b", h):
        return "kernel" if fn else "kernel_decl"
    if re.search(r"\b__device__\b", h):
        if fn:
            return "hostdevfn" if re.search(r"\b__host__\b", h) else "devfn"
        if "(" in h and it.kind == "decl" and not re.search(r"\b__constant__\b|=", h):
            return "devfn_decl"
        return "devvar"
    if re.search(r"\b__constant__\b", h):
        return "devvar"
    return "other"


# ---------------------------------------------------------------- signatures
def parse_signature(header):
    """(template_params, name, params, qualifiers) of a function header, header ending at its ')'."""
    h = header.strip()
    tparams = None
    tm = re.match(r"template\s*<", h)
    if tm:
        close = match_angle(h, tm.end() - 1)
        tparams = h[tm.end():close].strip()
        h = h[close + 1:].strip()
    h = re.sub(r"__launch_bounds__\s*\([^()]*(\([^()]*\)[^()]*)*\)", " ", h)
    # the parameter list: the last top-level (...) of the header
    close = len(h) - 1
    while h[close] != ")":
        close -= 1
    depth = 0
    k = close
    while k >= 0:
        if h[k] == ")":
            depth += 1
        elif h[k] == "(":
            depth -= 1
            if depth == 0:
                break
        k -= 1
    params = h[k + 1:close]
    before = h[:k].strip()
    nm = re.search(r"([A-Za-z_]\w*)\s*$", before)
    name = nm.group(1)
    ret = before[:nm.start()]
    return tparams, name, params, ret


def parse_params(params):
    """[(type, name, default)] of a parameter list."""
    out = []
    for p in split_commas(params):
        p = p.strip()
        if not p or p == "void":
            continue
        default = None
        dm = None
        depth = 0
        for i, c in enumerate(p):
            if c in "(<[":
                depth += 1
            elif c in ")>]":
                depth -= 1
            elif c == "=" and depth == 0:
                dm = i
                break
        if dm is not None:
            default = p[dm + 1:].strip()
            p = p[:dm].strip()
        nm = re.search(r"([A-Za-z_]\w*)\s*(\[[^\]]*\])?\s*$", p)
        if not nm:
            out.append((p, None, default))
            continue
        out.append((p[:nm.start()].strip() + (nm.group(2) or ""), nm.group(1), default))
    return out


def parse_tparams(tparams):
    """[(kind, type, name, default)]: kind 'type' for typename/class, 'value' otherwise."""
    out = []
    if not tparams:
        return out
    for p in split_commas(tparams):
        p = p.strip()
        default = None
        if "=" in p:
            p, default = p.split("=", 1)
            p, default = p.strip(), default.strip()
        m = re.match(r"(typename|class)\s+(\w+)$", p)
        if m:
            out.append(("type", None, m.group(2), default))
            continue
        m = re.match(r"(.*?)\s*\b(\w+)$", p)
        out.append(("value", m.group(1).strip(), m.group(2), default))
    return out


# ---------------------------------------------------------------- device-side text rewriting
TYPE_MAP = [
    (r"\bunsigned\s+long\s+long(\s+int)?\b", "ulong"),
    (r"\blong\s+long(\s+int)?\b", "long"),
    (r"\bunsigned\s+long\b", "ulong"),
    (r"\bunsigned\s+int\b", "uint"),
    (r"\bunsigned\s+short\b", "ushort"),
    (r"\bunsigned\s+char\b", "uchar"),
    (r"\bsigned\s+char\b", "char"),
    (r"\bunsigned\b(?!\s*(int|char|short|long))", "uint"),
    (r"\bsize_t\b", "ulong"),
    (r"\bptrdiff_t\b", "long"),
    (r"\b__nv_bfloat162\b", "lf_bf162"),
    (r"\b__nv_bfloat16\b", "lf_bf16"),
    (r"\b__half2\b", "half2"),
    (r"\b__half\b", "half"),
    (r"\b__restrict__\b", ""),
    (r"\b__forceinline__\b", ""),
    (r"\b__noinline__\b", ""),
    (r"\b__host__\b", ""),
    (r"\b__device__\b", ""),
    (r"__launch_bounds__\s*\([^()]*(\([^()]*\)[^()]*)*\)", ""),
    (r"\b__align__\s*\(", "alignas("),
    (r"\bstd::", "metal::"),
]

PTR_DECL = re.compile(
    r"(?P<lead>(^|[;{}(]|\bfor\s*\()\s*)(?P<const1>const\s+)?(?P<type>[A-Za-z_][\w:]*(\s*<[^;(){}]*?>)?)\s*(?P<const2>const\s*)?\*\s*(const\s+)?(__restrict__\s+)?(?P<name>[A-Za-z_]\w*)\s*=(?!=)",
    re.M)


def device_ptr_type(t):
    """A pointer parameter's type with Metal address spaces: every pointee in device memory."""
    t = t.strip()
    if "*" not in t and "&" not in t:
        return t
    if t.endswith("&"):
        return "thread " + t
    base, stars = t.split("*", 1)
    out = "device " + base.strip() + "*"
    for s in stars.split("*")[:-1] if "*" in stars else []:
        out += " device" + (" " + s.strip() if s.strip() else "") + "*"
    rest = stars.split("*")[-1].strip() if "*" in stars else stars.strip()
    if rest:
        out += " " + rest
    return out


def rewrite_casts(text):
    """reinterpret_cast<T*>(e) and C-style (T*)e as lf_rcast<T>(e), which keeps e's address space."""
    out = text
    # reinterpret_cast<...*>( and const_cast
    def repl_rc(m):
        return f"lf_rcast<{m.group(1).strip()}>("
    out = re.sub(r"\breinterpret_cast\s*<\s*([^<>*]*(<[^<>]*>)?[^<>*]*)\*\s*>\s*\(", repl_rc, out)
    out = re.sub(r"\breinterpret_cast\s*<\s*([^<>&]*)&\s*>\s*\(", lambda m: f"*lf_rcast<{m.group(1).strip()}>(&", out)
    # C-style pointer casts: (T*)expr, (const T*)expr -> lf_rcast<T>(expr)
    res = []
    i = 0
    cre = re.compile(r"\(\s*((const\s+)?(volatile\s+)?[A-Za-z_]\w*(\s*<[^()<>]*>)?(\s+const)?)\s*\*\s*\)")
    while True:
        m = cre.search(out, i)
        if not m:
            res.append(out[i:])
            break
        # not a function call or declaration like "f(float*)": a cast is preceded by an operator or open bracket
        prev = out[:m.start()].rstrip()
        if prev and (prev[-1].isalnum() or prev[-1] in "_)]"):
            res.append(out[i:m.end()])
            i = m.end()
            continue
        res.append(out[i:m.start()])
        j = m.end()
        while j < len(out) and out[j] == " ":
            j += 1
        start = j
        # the operand: a unary expression
        while j < len(out) and out[j] in "&*!~-+":
            j += 1
        if j < len(out) and out[j] == "(":
            j = match(out, j, "(", ")") + 1
        else:
            mm = re.match(r"[A-Za-z_0-9.]+", out[j:])
            j += mm.end() if mm else 0
        while j < len(out):
            if out[j] == "[":
                j = match(out, j, "[", "]") + 1
            elif out[j] == "(" and out[j - 1] not in " ":
                j = match(out, j, "(", ")") + 1
            elif out.startswith("->", j) or out[j] == ".":
                j += 2 if out[j] == "-" else 1
                mm = re.match(r"[A-Za-z_]\w*", out[j:])
                j += mm.end() if mm else 0
            else:
                break
        res.append(f"lf_rcast<{m.group(1).strip()}>({out[start:j]})")
        i = j
    return "".join(res)


LAMBDA_HELPERS = []      # helper functions lowered from lambdas, emitted before the function that held them
KEYWORDS = set("""if else for while do return break continue switch case default sizeof const constexpr auto int float
uint bool char short long ulong ushort uchar half half2 float2 float3 float4 int2 int3 int4 uint2 uint4 void true false
static_cast reinterpret_cast struct template typename thread device threadgroup constant inline unsigned signed
INFINITY nullptr""".split())


def lower_lambdas(body, ctx_names=()):
    """`auto f = [...](params) { return expr; };` as a macro; statement bodies as a statement expression; a body
    that returns from the middle as a helper FUNCTION (LAMBDA_HELPERS) taking what it captures by value.
    Returns (body, macro names) - the caller #undefs them after the function."""
    names = []
    lam = re.compile(r"\bauto\s+(\w+)\s*=\s*\[[^\]]*\]\s*\(")
    while True:
        m = lam.search(body)
        if not m:
            break
        name = m.group(1)
        po = m.end() - 1
        pc = match(body, po, "(", ")")
        params = parse_params(body[po + 1:pc])
        k = pc + 1
        ret = None
        rm = re.match(r"\s*(mutable\s*)?(->\s*([^{]+))?\s*\{", body[k:])
        if not rm:
            raise ValueError(f"lambda {name}: no body")
        ret = rm.group(3)
        bo = k + rm.end() - 1
        bc = match(body, bo, "{", "}")
        inner = strip_comments(body[bo + 1:bc]).strip()
        semi = body.find(";", bc)
        # substitute parameters
        def subst(text):
            for typ, pname, _ in params:
                if pname is None:
                    continue
                if "&" in typ or "*" in typ or typ.strip() in ("auto", "const auto"):
                    rep = f"({pname}_lfarg)"
                else:
                    rep = f"(({typ.replace('const ', '').strip()})({pname}_lfarg))"
                text = re.sub(rf"\b{pname}\b", rep, text)
            return text
        args = ", ".join(f"{p[1]}_lfarg" for p in params if p[1])
        rets = re.findall(r"\breturn\b", inner)
        sm = re.fullmatch(r"return\s+(.*?);?", inner, re.S)
        if sm and len(rets) == 1:
            macro = f"({subst(sm.group(1))})"
            if ret:
                macro = f"(({ret.strip()})({macro}))"
        elif len(rets) > 1 or (rets and not re.search(r"return\s+([^;]*);\s*$", inner)):
            # a helper function: the captured names (declared before the lambda, used in it) passed by value
            before = body[:m.start()]
            declared = live_names(before) | set(ctx_names)
            used_ids = set(re.findall(r"\b([A-Za-z_]\w*)\b", inner))
            pnames = {p[1] for p in params if p[1]}
            caps = sorted((declared & used_ids) - pnames - KEYWORDS)
            fname = f"_lf_lambda_{name}_{len(LAMBDA_HELPERS)}"
            tps = [f"typename _C{i}" for i in range(len(caps))]
            pps = []
            for pi, (t, n, _) in enumerate(params):
                if not n:
                    continue
                if "*" in t:                          # (a pointer parameter: any address space)
                    tps.append(f"typename _LP{pi}")
                    pps.append(f"_LP{pi} {n}")
                else:
                    pps.append(f"{'thread ' if '&' in t else ''}{t} {n}")
            fps = ["thread const LfCtx& _c"] + [f"_C{i} {c}" for i, c in enumerate(caps)] + pps
            tmpl = f"template <{', '.join(tps)}>\n" if tps else ""
            LAMBDA_HELPERS.append(f"{tmpl}inline auto {fname}({', '.join(fps)}) {{{inner}}}\n")
            args_ = ", ".join(["_c"] + caps + [f"{p[1]}_lfarg" for p in params if p[1]])
            body = body[:m.start()] + f"\n#define {name}({args}) {fname}({args_})\n" + body[semi + 1:]
            names.append(name)
            continue
        else:
            # parameters as locals (a statement body may assign them), through a temporary so an argument that
            # names an outer variable of the same name reads the outer one
            copies, refs = "", []
            for typ, pname, _ in params:
                if pname is None:
                    continue
                if "&" in typ or "*" in typ:
                    refs.append((typ, pname))
                    continue
                t = typ.replace("const ", "").strip()
                t = "auto" if t == "auto" else t
                copies += f"{t} _lft_{pname} = ({pname}_lfarg); "
            for typ, pname, _ in params:
                if pname and not ("&" in typ or "*" in typ):
                    copies += f"{typ.replace('const ', '').strip()} {pname} = _lft_{pname}; "
            def subst_refs(text):
                for typ, pname in refs:
                    text = re.sub(rf"\b{pname}\b", f"({pname}_lfarg)", text)
                return text
            if rets:
                lm = re.search(r"return\s+([^;]*);\s*$", inner)
                stmts, last = inner[:lm.start()], lm.group(1)
                macro = "({ " + copies + subst_refs(stmts) + f" {subst_refs(last)}; }})"
            else:
                macro = "({ " + copies + subst_refs(inner) + " })"
        macro = re.sub(r"#\s*pragma\s+unroll[^\n]*", "_Pragma(\"unroll\")", macro)
        macro = macro.replace("\n", " ")
        body = body[:m.start()] + f"\n#define {name}({args}) {macro}\n" + body[semi + 1:]
        names.append(name)
    return body, names


def shared_in_place(body):
    """`__shared__` as `threadgroup` where it stands (Metal takes threadgroup declarations anywhere in a kernel);
    `extern __shared__ T x[];` as a pointer into the dynamic threadgroup memory."""
    def repl(m):
        text = m.group(0)
        if text.lstrip().startswith("extern"):
            em = re.match(r"\s*extern\s+__shared__\s+(alignas\s*\([^)]*\)\s*|__align__\s*\([^)]*\)\s*)?([\w\s<>:,]+?)\s+(\w+)\s*\[\s*\]\s*;", text)
            if not em:
                raise ValueError("extern __shared__ of an unexpected form: " + text)
            typ, name = em.group(2).strip(), em.group(3)
            return f"threadgroup {typ}* {name} = (threadgroup {typ}*)_lf_smem;"
        return re.sub(r"\b__shared__\b", "threadgroup", text)
    return re.sub(r"(extern\s+)?__shared__[^;]*;", repl, body)


def struct_members(text):
    """A struct for device code: its pointer MEMBERS (declarations at the struct's own brace depth, not expressions
    in its methods) point into device memory, and a double member keeps CUDA's 8 bytes."""
    m = mask(text)
    out, depth, start = [], 0, 0
    segs = []        # (start, end) of depth-1 text
    i = 0
    seg_start = None
    for i, c in enumerate(m):
        if c == "{":
            if depth == 0:
                seg_start = i + 1
            elif depth == 1:
                segs.append((seg_start, i))
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 1:
                seg_start = i + 1
            elif depth == 0:
                segs.append((seg_start, i))
    res, last = [], 0
    for a, b in segs:
        res.append(text[last:a])
        seg = text[a:b]
        seg = re.sub(r"((?:^|[;{}])\s*)((const\s+)?[A-Za-z_][\w:<>]*\s*\*)", lambda mm: mm.group(1) + "device " + mm.group(2), seg)
        seg = re.sub(r"((?:^|[;{}])\s*)double\b", lambda mm: mm.group(1) + "lf_f64", seg)
        res.append(seg)
        last = b
    res.append(text[last:])
    return "".join(res)


def local_references(body, shared, pointers):
    """`T& r = base[...]` with an address space Metal requires: threadgroup if base is __shared__, device if it is a
    pointer into a buffer, thread otherwise."""
    def repl(m):
        base = m.group(4)
        if m.group(2).strip() in ("auto", "const auto") or re.search(r"\b(thread|device|threadgroup|constant)\s*$", m.group(1)):
            return m.group(0)
        space = "threadgroup" if base in shared else ("device" if base in pointers else "thread")
        return f"{m.group(1)}{space} {m.group(2)}& {m.group(3)} = {base}"
    return re.sub(r"((?:^|[;{}])\s*)((?:const\s+)?[A-Za-z_][\w:<>]*)\s*&\s*([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)(?=\s*\[)", repl, body, flags=re.M)


def vector_element_casts(text):
    """`*lf_rcast<T>(&v.x)` - a reinterpretation of a vector's element, whose address Metal will not take - as a
    by-value reinterpretation."""
    return re.sub(r"\*\s*lf_rcast<([^<>]+)>\(\s*&\s*([A-Za-z_][\w\[\]]*\.[xyzw])\s*\)", r"lf_reinterp<\1>(\2)", text)


def auto_pointers(text):
    """`T* p = e;` as `auto p = e;`: a local pointer keeps the address space of what it was set from."""
    def repl(m):
        t = m.group("type")
        if t in ("return", "else", "case", "goto", "delete", "new", "throw", "sizeof", "typedef", "using"):
            return m.group(0)
        return f"{m.group('lead')}auto {m.group('name')} ="
    prev = None
    while prev != text:
        prev = text
        text = PTR_DECL.sub(repl, text)
    return text


CTX_RE = re.compile(r"\b(threadIdx|blockIdx|blockDim|gridDim|_lf_smem)\b")


def needs_ctx_shfl(body):
    """A narrow-width shuffle (a 4th argument) needs the lane, which only the context carries."""
    for m in re.finditer(r"\b__shfl_(down|up|sync|xor)_sync\s*\(", body):
        try:
            close = match(body, m.end() - 1, "(", ")")
        except ValueError:
            continue
        if len(split_commas(body[m.end():close])) >= 4:
            return True
    return False


# ---------------------------------------------------------------- the translation of one port
class Port:
    def __init__(self, root, out, overrides):
        self.root = os.path.abspath(root)
        self.out = os.path.abspath(out)
        self.overrides = overrides
        self.files = []          # CUDA sources in include order (each once)
        self.kernels = []        # (file, name, tparams, params, mslname, structname, index)
        self.device_items = []   # (file, text) in order
        self.problems = []

    def source(self, path):
        return self.parsed[path][0] if hasattr(self, "parsed") and path in self.parsed else open(path).read()

    def walk(self, path, seen):
        path = os.path.abspath(path)
        if path in seen:
            return
        seen.add(path)
        src = open(path).read()
        for inc in re.findall(r'^\s*#\s*include\s+"([^"]+)"', src, re.M):
            p = os.path.normpath(os.path.join(os.path.dirname(path), inc))
            if p.startswith(CUDA) and (p.endswith(".cuh") or p.endswith(".cu")) and os.path.exists(p):
                self.walk(p, seen)
        self.files.append(path)

    def run(self):
        self.walk(self.root, set())
        self.parsed = {}
        self.port = os.path.basename(os.path.dirname(os.path.dirname(self.root)))
        # A port's changes to the CUDA sources sit flat in metal/<owner>/, named by what they change:
        #   <function>.host.h    a host function replaced
        #   <kernel>.kernel.cu   a kernel's body replaced (below, where kernels are emitted)
        #   <file>.patch         exact old -> new blocks, and `insert` blocks placed after the file's includes
        #   <file>               the whole file replaced
        # They belong to the port that OWNS the CUDA file (cuda/<owner>/...): cuda/af2 includes cuda/af3's headers, and
        # metal/af3's changes to those apply to it too.
        self.host_overrides = {}       # (owner, function name) -> text
        for owner in sorted(os.listdir(os.path.join(REPO, "metal", "legacy"))):
            odir = os.path.join(REPO, "metal", "legacy", owner)
            if os.path.isdir(odir):
                for f in sorted(os.listdir(odir)):
                    if f.endswith(".host.h"):
                        self.host_overrides[(owner, f[:-len(".host.h")])] = open(os.path.join(odir, f)).read()
        for path in self.files:
            src = open(path).read()
            owner = os.path.relpath(path, CUDA).split(os.sep)[0]
            here = os.path.join("metal", "legacy", owner, os.path.basename(path))
            if os.path.exists(os.path.join(REPO, here)):     # (a hand-written Metal version of the whole file)
                src = open(os.path.join(REPO, here)).read()
            if os.path.exists(os.path.join(REPO, here + ".patch")):
                text = open(os.path.join(REPO, here + ".patch")).read()
                for blk in re.finditer(r"^@@@ (old|insert)\n(.*?)^(?:@@@ new\n(.*?))?^@@@ end\n", text, re.M | re.S):
                    if blk.group(1) == "insert":       # after the file's own includes, before anything that may call it
                        last = 0
                        for im in re.finditer(r"^\s*#\s*include\b[^\n]*\n", src, re.M):
                            last = im.end()
                        src = src[:last] + f"// ---- inserted by {here}.patch\n" + blk.group(2) + "// ---- end\n" + src[last:]
                        continue
                    old, new = blk.group(2), blk.group(3) or ""
                    if src.count(old) != 1:            # (the CUDA source moved: stop, and say which block)
                        raise SystemExit(f"{here}.patch: a block matches {src.count(old)} times in {path}, not once:\n{old}")
                    src = src.replace(old, new, 1)
            m = mask(src)
            items = items_of(m, 0, len(m))
            self.parsed[path] = (src, m, [(it, classify(it, m)) for it in items])
        host_out = {}
        for path in self.files:
            src, m, items = self.parsed[path]
            pieces, last = [], 0
            ns = []
            for it, kind in items:
                text = src[it.start:it.end]
                pieces.append(src[last:it.start])
                last = it.end
                if kind == "ns_open":
                    nm = re.search(r"namespace\s+([\w:]+)", it.header)
                    ns.append(nm.group(1) if nm else "")
                elif kind == "ns_close":
                    ns.pop()
                if kind == "kernel":
                    pieces.append(self.kernel(path, src, m, it, "".join(n + "::" for n in ns if n)))
                elif kind in ("devfn", "devfn_decl", "kernel_decl"):
                    pieces.append("")
                elif kind == "devvar":
                    pieces.append(re.sub(r"\b__constant__\b|\b__device__\b", "", text))
                elif kind == "other" and it.kind == "def" and "(" in it.header and self.host_overrides:
                    try:
                        hname = parse_signature(it.header)[1]
                    except Exception:
                        hname = None
                    owner = os.path.relpath(path, CUDA).split(os.sep)[0]
                    if (owner, hname) in self.host_overrides:
                        # (a host function of the Metal build's own, metal/<owner>/host/<name>.h: every overload of the
                        # name replaced by the file, once)
                        pieces.append(f"// {hname}: metal/{owner}/host/{hname}.h\n" + self.host_overrides.pop((owner, hname)))
                        self.replaced_host = getattr(self, "replaced_host", set()) | {hname}
                    elif hname in getattr(self, "replaced_host", set()):
                        pieces.append("")
                    else:
                        pieces.append(text)
                else:
                    pieces.append(text)
            pieces.append(src[last:])
            host_out[path] = self.rewrite_launches("".join(pieces))
        for path, host in host_out.items():
            rel = os.path.relpath(path, CUDA)
            dst = os.path.join(self.out, "gen", rel)
            # /dev/shm is Linux's: the system's temporary directory (lf::tmpDir, lfcuda.h)
            host = re.sub(r'"/dev/shm/([^"]*)"', r'(std::string(lf::tmpDir()) + "/\1")', host)
            # libzstd: on a Mac it is linked into the binary itself (metal/build.sh), where dlsym finds it
            host = re.sub(r'dlopen\("libzstd\.so\.1",\s*RTLD_NOW\)', "dlopen(nullptr, RTLD_NOW)", host)
            host = re.sub(r'dlopen\("libzstd\.so",\s*RTLD_NOW\)', "dlopen(nullptr, RTLD_NOW)", host)
            # std::to_chars for a float needs macOS 13.3's libc++: lf::to_chars (lfcuda.h) prints as it does
            host = re.sub(r"\bstd::to_chars\s*\(", "lf::to_chars(", host)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w") as f:
                f.write(f"// GENERATED by metal/tools/cu2metal.py from cuda/{rel} - do not edit\n#include \"lf_kernels.h\"\n" + host)
        self.write_device()
        self.write_table()
        # the plain C++ the ports include beside the CUDA (cuda/featurise): linked, not translated
        link = os.path.join(self.out, "gen", "featurise")
        if not os.path.exists(link):
            os.symlink(os.path.join(CUDA, "featurise"), link)

    # -- a kernel: its host stub, and its Metal twin queued
    def kernel(self, path, src, m, it, qual=""):
        header = src[it.start:it.start + len(it.header)]
        body_open = it.start + len(it.header)
        tparams, name, params, ret = parse_signature(header)
        ps = parse_params(params)
        tps = parse_tparams(tparams)
        index = len(self.kernels)
        msl = name
        struct = f"{name}__a{index}"
        # a hand-written body (metal/<port>/<name>.kernel.cu, CUDA syntax, translated as the original is) replaces
        # the CUDA one; the signature and the argument struct stay the translator's
        owner = os.path.relpath(path, CUDA).split(os.sep)[0]
        ofile = os.path.join(REPO, "metal", "legacy", owner, name + ".kernel.cu")
        override = os.path.exists(ofile)
        obody = None
        if override:
            otext = open(ofile).read()
            om = mask(otext)
            ob = om.find("{")
            obody = otext[ob:match(om, ob, "{", "}") + 1]
        self.kernels.append(dict(file=path, name=name, tps=tps, ps=ps, msl=msl, struct=struct, index=index, qual=qual,
                                 body=obody or src[body_open:it.end], mbody=mask(obody) if obody else m[body_open:it.end],
                                 orig=src[body_open:it.end], tparams=tparams, override=override))
        # the host stub: the same template header and parameters
        targs = []
        for kind, typ, tn, _ in tps:
            targs.append(f"lf::tname<{tn}>()" if kind == "type" else f"lf::tval<{typ}>({tn}, \"{typ}\")")
        tmpl = f"template <{tparams}>\n" if tparams is not None else ""
        # (each argument as the kernel declares it - a default-promoted or converted value in the declared type)
        adds = "".join(f"  _lfa.add(({re.sub(r'__restrict__', '', p[0]).strip()})({p[1]}));\n" if "[" not in p[0] else f"  _lfa.add({p[1]});\n" for p in ps if p[1])
        # (parameter text as written, defaults kept - the stub is called exactly as the kernel was)
        return (f"{tmpl}inline void {name}({params}) {{\n  lf::KArgs _lfa;\n{adds}"
                f"  _lfa.finish();\n  lf::launch({index}, {{{', '.join(targs)}}}, _lfa);\n}}")

    def rewrite_launches(self, text):
        m = mask(text)
        out, last = [], 0
        for lm in re.finditer(r"<<<", m):
            s = lm.start()
            if s < last:
                continue
            # the callee, backwards: identifier, optional template args, optional ::
            k = s - 1
            while k >= 0 and m[k].isspace():
                k -= 1
            if m[k] == ">":
                depth = 0
                while k >= 0:
                    if m[k] == ">":
                        depth += 1
                    elif m[k] == "<":
                        depth -= 1
                        if depth == 0:
                            break
                    k -= 1
                k -= 1
            while k >= 0 and (m[k].isalnum() or m[k] in "_:"):
                k -= 1
            callee_start = k + 1
            # the config, up to >>> at paren depth 0
            j = lm.end()
            depth = 0
            while True:
                if m[j] == "(":
                    depth += 1
                elif m[j] == ")":
                    depth -= 1
                elif m.startswith(">>>", j) and depth == 0:
                    break
                j += 1
            cfg = text[lm.end():j]
            a = j + 3
            while m[a].isspace():
                a += 1
            ac = match(m, a, "(", ")")
            callee = text[callee_start:s].strip()
            out.append(text[last:callee_start])
            out.append(f"(lf::setLaunch({cfg}), {callee}{text[a:ac + 1]})")
            last = ac + 1
        out.append(text[last:])
        return "".join(out)

    # -- the device side
    def write_device(self):
        # every device function, and which cannot be translated: inline PTX, a lambda passed as an argument
        fns = {}
        for path in self.files:
            src, m, items = self.parsed[path]
            for it, kind in items:
                if kind in ("devfn", "hostdevfn"):
                    header = src[it.start:it.start + len(it.header)]
                    try:
                        name = parse_signature(header)[1]
                    except Exception:
                        continue
                    fns.setdefault(name, []).append((path, it, m[it.start:it.end]))
        def calls(body, names):
            return {n for n in names if re.search(rf"\b{n}\s*(<[^;{{}}]*?>)?\s*\(", body)}
        def intrinsic_bad(body):
            if re.search(r"\basm\b", body):
                return "inline PTX"
            if re.search(r"[(,]\s*\[[&=]?[\w, &]*\]\s*\(", body):
                return "a lambda passed as an argument"
            return None
        self.bad_fns = {}
        for name, defs_ in fns.items():
            for path, it, body in defs_:
                r = intrinsic_bad(body)
                if r:
                    self.bad_fns[name] = r
        changed = True
        while changed:
            changed = False
            for name, defs_ in fns.items():
                if name in self.bad_fns:
                    continue
                for path, it, body in defs_:
                    c = calls(body, self.bad_fns)
                    if c:
                        self.bad_fns[name] = "calls " + sorted(c)[0]
                        changed = True
                        break
        for k in self.kernels:
            r = intrinsic_bad(k["mbody"])
            c = calls(k["mbody"], self.bad_fns)
            if r or c:
                k["untranslated"] = r or ("calls " + sorted(c)[0])
        # the identifiers translated device code uses, and the declarations that define them (a closure)
        used = set()
        for name, defs_ in fns.items():
            if name not in self.bad_fns:
                for path, it, body in defs_:
                    used |= set(re.findall(r"[A-Za-z_]\w*", body))
        for k in self.kernels:
            if not k.get("untranslated"):
                used |= set(re.findall(r"[A-Za-z_]\w*", k["mbody"]))
                for p in k["ps"]:
                    used |= set(re.findall(r"[A-Za-z_]\w*", p[0]))
                for _, typ, _, d in k["tps"]:
                    used |= set(re.findall(r"[A-Za-z_]\w*", (typ or "") + " " + (d or "")))
        def defines(text):
            names = set()
            mt = mask(text)
            mt = re.sub(r"^\s*template\s*<[^{;]*?>\s*(?=(struct|class|union|using|constexpr|inline|static|const|[A-Za-z_]))", "", mt)
            head = mt.split("{")[0]
            if "(" in head and not re.search(r"\b(struct|class|union|enum)\b", head.split("(")[0]) and "=" not in head.split("(")[0]:
                return names
            m1 = re.search(r"\b(struct|class|union|enum(\s+class)?)\s+(\w+)", mt)
            if m1:
                names.add(m1.group(3))
                if m1.group(1).startswith("enum"):
                    br = mt.find("{")
                    if br >= 0:
                        names |= set(re.findall(r"(\w+)\s*(?:=[^,}]*)?[,}]", mt[br + 1:]))
                return names
            m2 = re.match(r"\s*using\s+(\w+)\s*=\s*([\w:]+)", mt)
            if m2:
                if m2.group(1) not in ("half", "half2") and not m2.group(2).startswith("std::"):
                    names.add(m2.group(1))
                return names
            m3 = re.search(r"\btypedef\b.*?(\w+)\s*;\s*$", mt, re.S)
            if m3:
                names.add(m3.group(1))
                return names
            if re.search(r"\b(constexpr|const)\b", mt) and "(" not in mt.split("=")[0]:
                if re.search(r"\bstd::|\bstring\b", mt.split("=")[0]):
                    return names
                # every declarator: `constexpr int A = 1, B = 2;`
                for decl in split_commas(mt.rstrip().rstrip(";")):
                    m4 = re.search(r"(\w+)\s*(\[[^\]]*\])?\s*=", decl)
                    if m4:
                        names.add(m4.group(1))
            return names
        cands = []
        for path in self.files:
            src, m, items = self.parsed[path]
            for it, kind in items:
                if kind in ("other", "devvar"):
                    text = src[it.start:it.end]
                    cands.append(((path, it.start), text, defines(text) if kind == "other" else
                                  set(re.findall(r"(\w+)\s*(\[[^\]]*\])*\s*(=|;)", mask(text))[:1] and
                                      [re.findall(r"(\w+)\s*(?:\[[^\]]*\])*\s*(?:=|;)", mask(text))[0]])))
        chosen = set()
        changed = True
        while changed:
            changed = False
            for key, text, names in cands:
                if key in chosen or not names or not (names & used):
                    continue
                chosen.add(key)
                used |= set(re.findall(r"[A-Za-z_]\w*", mask(text)))
                changed = True
        # which device functions need the context (directly or through a call)
        needs = set()
        changed = True
        while changed:
            changed = False
            for name, defs_ in fns.items():
                if name in needs or name in self.bad_fns:
                    continue
                for path, it, body in defs_:
                    if CTX_RE.search(body) or needs_ctx_shfl(body) or calls(body, needs):
                        needs.add(name)
                        changed = True
                        break
        self.ctx_funcs = needs
        self.fn_defs = fns
        out = ["// GENERATED by metal/tools/cu2metal.py - do not edit\n#include \"prelude.metal\"\n"]
        emitted = set()
        def emit(path):
            if path in emitted:
                return
            emitted.add(path)
            src, m, items = self.parsed[path]
            out.append(f"\n// ---- {os.path.relpath(path, CUDA)}\n")
            for it, kind in items:
                text = src[it.start:it.end]
                if kind == "pp":
                    im = re.match(r'\s*#\s*include\s+"([^"]+)"', text)
                    if im:
                        p = os.path.normpath(os.path.join(os.path.dirname(path), im.group(1)))
                        if p in self.parsed:
                            emit(p)
                        continue
                    if re.match(r"\s*#\s*(pragma\s+once|include)", text):
                        continue
                    out.append(self.dev_text(text) + "\n")
                elif kind in ("ns_open", "ns_close"):
                    out.append(text + "\n")
                elif kind in ("other", "devvar"):
                    if (path, it.start) in chosen:
                        out.append(self.dev_decl(text, kind) + "\n")
                elif kind in ("devfn", "hostdevfn"):
                    header = src[it.start:it.start + len(it.header)]
                    try:
                        name = parse_signature(header)[1]
                    except Exception:
                        continue
                    if name in self.bad_fns:
                        out.append(f"// {name}: not translated ({self.bad_fns[name]})\n")
                        continue
                    try:
                        out.append(self.dev_function(src, m, it) + "\n")
                    except Exception as e:
                        self.problems.append(f"{os.path.relpath(path, CUDA)}: device function {name}: {e}")
                elif kind == "kernel":
                    k = next(k for k in self.kernels if k["file"] == path and k["orig"] == src[it.start + len(it.header):it.end])
                    if k.get("untranslated"):
                        out.append(f"// {k['name']}: not translated ({k['untranslated']})\n")
                    else:
                        try:
                            out.append(self.dev_kernel(k) + "\n")
                        except Exception as e:
                            k["untranslated"] = str(e)
                            self.problems.append(f"{os.path.relpath(path, CUDA)}: kernel {k['name']}: {e}")
        emit(self.root)
        with open(os.path.join(self.out, "kernels.metal"), "w") as f:
            f.write("".join(out))

    def dev_text(self, text):
        for pat, rep in TYPE_MAP:
            text = re.sub(pat, rep, text)
        return text

    def dev_decl(self, text, kind):
        text = re.sub(r"\b__constant__\b", "constant", text)
        text = re.sub(r"\binline\s+constexpr\b", "constexpr", text)
        text = re.sub(r"\binline\s+const\b", "const", text)
        # program-scope constants live in the constant address space
        if re.match(r"\s*(static\s+)?(constexpr|const)\b", text) and not re.search(r"\b(struct|class|enum|using)\b", text):
            text = re.sub(r"^\s*(static\s+)?(constexpr|const)\b", r"constant constexpr" if "constexpr" in text.split("=")[0] else "constant", text, count=1)
        text = self.dev_text(text)
        # struct members: pointers into device memory; doubles as 8-byte storage
        if re.search(r"\b(struct|class)\b", text):
            text = struct_members(text)
        return text

    def ctx_calls(self, body):
        for n in self.ctx_funcs:
            body = re.sub(rf"\b({n}\s*(<[^;{{}}()]*?>)?\s*)\(\s*(\)?)",
                          lambda mm: f"{mm.group(1)}(_c" + (")" if mm.group(3) else ", "), body)
        return body

    def common_body(self, body, names=()):
        body = self.dev_text(body)
        # CUDA's `#pragma unroll` unrolls a constant-count loop fully; Metal's compiler takes the bare pragma as a hint,
        # and an array of matrix registers indexed by a loop it did not unroll goes to memory (5-10x slower)
        body = re.sub(r"#\s*pragma\s+unroll\s+(\d+)\s*$", r'_Pragma("clang loop unroll_count(\1)")', body, flags=re.M)
        body = re.sub(r"#\s*pragma\s+unroll\s*$", r'_Pragma("clang loop unroll(full)")', body, flags=re.M)
        body, lambdas = lower_lambdas(body, names)
        body = rewrite_casts(body)
        body = vector_element_casts(body)
        body = auto_pointers(body)
        body = self.ctx_calls(body)
        # double pointers stay 8-byte storage; double values are float
        body = re.sub(r"\bdouble\s*\*", "lf_f64*", body)
        body = re.sub(r"\bdouble\b", "float", body)
        body = re.sub(r"\bprintf\s*\(", "lf_printf(", body)
        body = re.sub(r"\bsincos\s*\((?=[^;]*,[^;]*,[^;]*\))", "lf_sincos(", body)
        return body, lambdas

    def dev_function(self, src, m, it):
        header = src[it.start:it.start + len(it.header)]
        body = src[it.start + len(it.header):it.end]
        tparams, name, params, ret = parse_signature(header)
        ps = parse_params(params)
        tps = parse_tparams(tparams) if tparams else []
        extra = []
        newps = []
        overloaded = len(self.fn_defs.get(name, [])) > 1
        for i, (typ, pname, default) in enumerate(ps):
            t = self.dev_text(typ)
            if "*" in t:
                tn = f"_LfP{i}"
                extra.append(f"typename {tn}")
                if overloaded and t.count("*") == 1:
                    elem = re.sub(r"\b(const|volatile)\b", "", t.split("*")[0]).strip()
                    elem = re.sub(r"\bdouble\b", "lf_f64", elem)
                    extra.append(f"LF_IS({tn}, {elem})")
                newps.append(f"{tn} {pname}")
            elif "&" in t:
                newps.append(f"thread {t} {pname}" + (f" = {default}" if default else ""))
            else:
                newps.append(f"{t} {pname}" + (f" = {default}" if default else ""))
        if name in self.ctx_funcs:
            newps.insert(0, "thread const LfCtx& _c")
        allt = ([tparams] if tparams else []) + extra
        tmpl = f"template <{', '.join(allt)}>\n" if allt else ""
        ret = self.dev_text(ret)
        ret = re.sub(r"\b(static|extern)\b", "", ret)
        ret = re.sub(r"\bdouble\b", "float", ret)
        if "*" in ret:
            ret = "auto"
        body, lambdas = self.common_body(body, [p[1] for p in ps if p[1]])
        undef = "".join(f"\n#undef {n}" for n in lambdas)
        helpers = "".join(LAMBDA_HELPERS); LAMBDA_HELPERS.clear()
        return f"{helpers}{tmpl}inline {ret.replace('inline', '').strip()} {name}({', '.join(newps)}) {body}{undef}"

    def dev_kernel(self, k):
        tps = k["tps"]
        tparams = k["tparams"] if k["tparams"] is not None else "int _lf_z"
        targs = ", ".join(t[2] for t in tps) if tps else "_lf_z"
        members, locals_ = [], []
        for typ, pname, _ in k["ps"]:
            t = self.dev_text(typ)
            if "*" in t:
                t = re.sub(r"\bdouble\b", "lf_f64", t)
                mt = device_ptr_type(t)
                if t.count("*") > 1:
                    # (a pointer to pointers: Metal takes none in a kernel's argument struct, so its address travels as
                    # an integer and is cast back here)
                    members.append(f"  ulong {pname};")
                    locals_.append(f"  {mt} {pname} = ({mt})_lf_a.{pname};")
                else:
                    members.append(f"  {mt} {pname};")
                    locals_.append(f"  {mt} {pname} = _lf_a.{pname};")
            elif re.search(r"\bdouble\b", t):
                members.append(f"  lf_f64 {pname};")
                locals_.append(f"  float {pname} = (float)_lf_a.{pname};")
            else:
                members.append(f"  {t} {pname};")
                locals_.append(f"  {t} {pname} = _lf_a.{pname};")
        body = shared_in_place(k["body"])
        body, lambdas = self.common_body(body, [p[1] for p in k["ps"] if p[1]] + [t[2] for t in k["tps"] if t[0] == "value"])
        shared = set(re.findall(r"\bthreadgroup\s+[\w:<> ]+?\s+(\w+)\s*\[", body)) | set(re.findall(r"threadgroup\s+[\w:<> ]+\*\s*(\w+)\s*=", body))
        pointers = {p[1] for p in k["ps"] if p[1] and "*" in p[0]}
        body = local_references(body, shared, pointers)
        undef = "".join(f"\n#undef {n}" for n in lambdas)
        helpers = "".join(LAMBDA_HELPERS); LAMBDA_HELPERS.clear()
        struct = helpers + f"template <{tparams}>\nstruct {k['struct']} {{\n" + "\n".join(members) + "\n};\n"
        sig = (f"template <{tparams}>\nkernel void {k['msl']}(constant {k['struct']}<{targs}>& _lf_a [[buffer(0)]], "
               "threadgroup char* _lf_smem [[threadgroup(0)]], LF_KERNEL_BUILTINS) {\n  LF_KERNEL_CTX\n")
        inner = body.strip()
        assert inner.startswith("{") and inner.endswith("}")
        return struct + sig + "\n".join(locals_) + "\n  " + inner + undef + "\n}\n"

    def write_table(self):
        lines = ["// GENERATED by metal/tools/cu2metal.py - do not edit"]
        for k in self.kernels:
            nt = len(k["tps"])
            status = "2" if k.get("untranslated") else ("1" if k["override"] else "0")
            lines.append(f'{{"{k["qual"]}{k["msl"]}", "{k["qual"]}{k["struct"]}", {nt}, {status}}},   // {os.path.relpath(k["file"], CUDA)}' + (f' - {k["untranslated"]}' if k.get("untranslated") else ""))
        with open(os.path.join(self.out, "gen", "lf_kernels.inc"), "w") as f:
            f.write("\n".join(lines) + "\n")
        with open(os.path.join(self.out, "gen", "lf_kernels.h"), "w") as f:
            f.write("#pragma once\n#include \"lfcuda.h\"\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overrides", default="")
    a = ap.parse_args()
    overrides = set()
    if a.overrides and os.path.exists(a.overrides):
        for line in open(a.overrides):
            line = line.split("#", 1)[0].strip()
            if line:
                overrides.add(line)
    port = Port(a.root, a.out, overrides)
    port.run()
    print(f"{len(port.files)} files, {len(port.kernels)} kernels ({sum(k['override'] for k in port.kernels)} overridden, "
          f"{sum(bool(k.get('untranslated')) for k in port.kernels)} not translated)")
    for k in port.kernels:
        if k.get("untranslated"):
            print(f"  untranslated {k['name']}: {k['untranslated']}")
    for p in port.problems:
        print("  PROBLEM", p)


if __name__ == "__main__":
    main()
