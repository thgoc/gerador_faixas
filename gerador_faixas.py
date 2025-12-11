# calculador_faixas.py — Interface Streamlit para o gerador de faixas (versão final formatada e com autoexec)

import re
import math
import csv
import io
import streamlit as st

# -------------------- Funções permitidas --------------------
import math as _math
ALLOWED_FUNCS = {name: getattr(_math, name) for name in dir(_math) if not name.startswith("_")}
ALLOWED_FUNCS.update({"abs": abs, "max": max, "min": min})

# -------------------- Utilidades --------------------
def ordem_grandeza(x: float) -> float:
    ax = abs(x)
    if ax == 0:
        return 1.0
    return 10 ** math.floor(math.log10(ax))

def step_for_value(v: float, fator: float) -> float:
    return ordem_grandeza(v) * fator

def round_down_to_step(v: float, fator: float) -> float:
    s = step_for_value(v, fator)
    if s == 0:
        return v
    out = math.floor(v / s) * s
    return 0.0 if abs(out) < 1e-15 else out

def step_for_variable_max(vmax: float, fator: float) -> float:
    return ordem_grandeza(vmax) * fator

def round_down_to_var_step(v: float, vmax: float, fator: float) -> float:
    s = step_for_variable_max(vmax, fator)
    if s == 0:
        return v
    out = math.floor(v / s) * s
    return 0.0 if abs(out) < 1e-15 else out

def avaliar(expr: str, contexto: dict) -> float:
    safe_globals = {"__builtins__": {}}
    safe_locals = dict(ALLOWED_FUNCS)
    safe_locals.update(contexto)
    return float(eval(compile(expr, "<expr>", "eval"), safe_globals, safe_locals))

def parse_expressao(expr_bruta: str) -> str:
    expr = expr_bruta.strip()
    if "=" in expr:
        expr = expr.split("=")[-1].strip()
    return expr

def sanitize_identifier(name: str) -> str:
    s = re.sub(r"\W+", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "var"
    if re.match(r"^\d", s):
        s = "V_" + s
    return s

# -------------------- Arredondamento configurável --------------------
def fmt_num_custom(x: float) -> str:
    usar_casas = st.session_state.get("usar_casas_decimais", False)
    casas = st.session_state.get("casas_decimais", 2)
    if usar_casas:
        return f"{round(x, casas):.{casas}f}"
    else:
        return f"{x:.10g}"

def arredondar_saida(y: float, fator: float) -> float:
    usar_casas = st.session_state.get("usar_casas_decimais", False)
    casas = st.session_state.get("casas_decimais", 2)
    if usar_casas:
        return round(y, casas)
    else:
        return round_down_to_step(y, fator)

# -------------------- Parser de inequação (.txt) --------------------
NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
TERM_RE = re.compile(r"\(\s*([^)]+?)\s*-\s*(" + NUM + r")\s*\)\s*\*\s*(" + NUM + r")")

def parse_ineq_txt_and_build_expr(txt: str):
    content = " ".join([ln.strip() for ln in txt.splitlines() if ln.strip()])
    parts = re.split(r"(?:<=|≤)", content, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("Não encontrei o operador '≤' ou '<=' na inequação.")
    lhs, rhs = parts[0].strip(), parts[1].strip()
    vars_order = []

    def repl(m: re.Match) -> str:
        raw = m.group(1).strip()
        const_ = m.group(2)
        coef_ = m.group(3)
        san = sanitize_identifier(raw)
        if san not in vars_order:
            vars_order.append(san)
        return f"({san} - {const_})*{coef_}"

    rhs_sanitized = TERM_RE.sub(repl, rhs)
    return rhs_sanitized, vars_order

# -------------------- Núcleo --------------------
def gerar_niveis_por_variavel(k: int, vmin: float, vmax: float, fator: float):
    if k <= 0:
        vmax_rd = round_down_to_var_step(vmax, vmax, fator)
        return [vmax_rd], 0, vmax_rd

    fracs = [i / k for i in range(k, 0, -1)]
    raw_vals = [vmin + f * (vmax - vmin) for f in fracs]
    vals_rd = [round_down_to_var_step(v, vmax, fator) for v in raw_vals]

    niveis = []
    for v in vals_rd:
        if not niveis or abs(niveis[-1] - v) > 1e-12:
            niveis.append(v)
    collapsed = k - len(niveis)
    vmax_rd = round_down_to_var_step(vmax, vmax, fator)
    if niveis and niveis[0] > vmax_rd + 1e-12:
        niveis[0] = vmax_rd
    return niveis, collapsed, vmax_rd

def faixa_label_val(var: str, idx: int, vals_desc: list[float], vmax_rd: float) -> str:
    k = len(vals_desc)
    top = vmax_rd if vmax_rd >= vals_desc[0] - 1e-12 else vals_desc[0]
    if k == 1:
        return f"≤ {fmt_num_custom(top)}"
    if idx == 0:
        low = vals_desc[1] if k >= 2 else vals_desc[0]
        return f"{fmt_num_custom(low)} < {var} ≤ {fmt_num_custom(top)}"
    if idx == k - 1:
        return f"≤ {fmt_num_custom(vals_desc[-1])}"
    low = vals_desc[idx + 1]
    high = vals_desc[idx]
    return f"{fmt_num_custom(low)} < {var} ≤ {fmt_num_custom(high)}"

def ordenar_variaveis_por_maximo(var_defs, expr, niveis_por_var, vmax_rd):
    nomes = [v["nome"] for v in var_defs]
    scores = []
    for v in nomes:
        max_y = -float("inf")
        for val in niveis_por_var[v]:
            contexto = {u: vmax_rd[u] for u in nomes}
            contexto[v] = val
            y = avaliar(expr, contexto)
            if y > max_y:
                max_y = y
        scores.append((v, max_y))
    scores.sort(key=lambda t: t[1], reverse=True)
    return [v for v, _ in scores]

def construir_tabela_linear(ordered_vars, niveis_por_var):
    ks = [len(niveis_por_var[v]) for v in ordered_vars]
    total_rows = math.prod(ks)
    blocks = []
    prod_right = 1
    for i in range(len(ks) - 1, -1, -1):
        blocks.append(prod_right)
        prod_right *= ks[i]
    blocks = list(reversed(blocks))

    linhas_vals = []
    for r in range(total_rows):
        row_vals = []
        for i, v in enumerate(ordered_vars):
            j = (r // blocks[i]) % ks[i]
            row_vals.append(niveis_por_var[v][j])
        linhas_vals.append(row_vals)
    return linhas_vals, ks, blocks

# -------------------- HTML & CSV --------------------
def gerar_html(ordered_vars, ks, blocks, niveis_por_var, vmax_rd, linhas_vals, expr, nome_saida, fator):
    css = """
    <style>
      :root { --green-dark:#7BA640; --green:#8FBF4D; --row-alt:#f7fbef; --border:#c9d9a5; }
      body { font-family: Arial, Helvetica, sans-serif; }
      .tbl { border-collapse: separate; border-spacing: 0; min-width: 980px; }
      .tbl th, .tbl td { border: 1px solid var(--border); padding: 10px 12px; text-align: center; }
      .tbl thead th { background: var(--green); color: #fff; font-weight: 700; }
      .tbl thead th:first-child { background: var(--green-dark); }
      .tbl tbody tr:nth-child(even) td { background: var(--row-alt); }
      .tbl td strong { font-weight: 700 !important; }
    </style>
    """
    html = ["<div>", css, "<table class='tbl'>"]
    html.append("<thead><tr>")
    for v in ordered_vars:
        html.append(f"<th>{v}</th>")
    html.append(f"<th>{nome_saida}</th></tr></thead><tbody>")

    for r in range(len(linhas_vals)):
        html.append("<tr>")
        for i, v in enumerate(ordered_vars):
            block = blocks[i]
            if r % block == 0:
                vals_desc = niveis_por_var[v]
                idx = (r // block) % ks[i]
                label = faixa_label_val(v, idx, vals_desc, vmax_rd[v])
                rowspan = block
                html.append(f"<td rowspan='{rowspan}'>{label}</td>")
        contexto = {v: linhas_vals[r][i] for i, v in enumerate(ordered_vars)}
        y = avaliar(expr, contexto)
        y_rd = arredondar_saida(y, fator)
        html.append(f"<td><strong>{fmt_num_custom(y_rd)}</strong></td>")
        html.append("</tr>")
    html.append("</tbody></table></div>")
    return "".join(html)

def gerar_csv_bytes(ordered_vars, niveis_por_var, vmax_rd, linhas_vals, expr, nome_saida, fator):
    output = io.StringIO()
    w = csv.writer(output, delimiter=";")
    w.writerow(ordered_vars + [nome_saida])
    for row_vals in linhas_vals:
        labels = []
        for v, val in zip(ordered_vars, row_vals):
            vals_desc = niveis_por_var[v]
            try:
                idx = vals_desc.index(val)
            except ValueError:
                idx = min(range(len(vals_desc)), key=lambda j: abs(vals_desc[j]-val))
            labels.append(faixa_label_val(v, idx, vals_desc, vmax_rd[v]))
        y = avaliar(expr, {v: row_vals[i] for i, v in enumerate(ordered_vars)})
        y_rd = arredondar_saida(y, fator)
        w.writerow(labels + [fmt_num_custom(y_rd)])
    return output.getvalue().encode("utf-8")

# -------------------- UI (Streamlit) --------------------
st.set_page_config(page_title="Gerador de Faixas", layout="wide")
st.title("🧮 Gerador de Faixas com Tabela Agrupada")

with st.sidebar:
    st.header("Parâmetros")
    txt_file = st.file_uploader("Carregar inequação (.txt)", type=["txt"])
    if st.button("📄 Ler inequação", use_container_width=True, disabled=(txt_file is None)):
        try:
            content = txt_file.read().decode("utf-8", errors="ignore")
            rhs_expr, vars_list = parse_ineq_txt_and_build_expr(content)
            st.session_state["expr_text"] = rhs_expr
            st.session_state["n_vars"] = max(1, len(vars_list))
            for i, nm in enumerate(vars_list):
                st.session_state[f"nome_{i}"] = nm
            st.success(f"Inequação carregada: {len(vars_list)} variável(is).")
        except Exception as e:
            st.error(f"Falha ao ler inequação: {e}")

    fator = st.number_input("Fator de arredondamento (0 < f ≤ 1)", min_value=1e-6, max_value=1.0,
                            step=0.01, format="%.5f", value=0.05, key="fator_val")
    usar_casas_decimais = st.checkbox("Usar arredondamento por casas decimais", value=False)
    casas_decimais = None
    if usar_casas_decimais:
        casas_decimais = st.number_input("Número de casas decimais", min_value=0, max_value=10, value=2, step=1)
    st.session_state["usar_casas_decimais"] = usar_casas_decimais
    st.session_state["casas_decimais"] = casas_decimais

    n = st.number_input("Número de variáveis", min_value=1, max_value=12,
                        value=st.session_state.get("n_vars", 3), step=1, key="n_vars")

st.markdown("Defina as variáveis e a expressão matemática.")

var_defs = []
cols = st.columns([1,1,1,1])
cols[0].markdown("**Nome**")
cols[1].markdown("**Mínimo**")
cols[2].markdown("**Máximo**")
cols[3].markdown("**Nº de faixas**")

default_names = ["PI", "PE", "Eixo", "X", "Y", "Z"]
n = st.session_state.get("n_vars", 3)

for i in range(n):
    c1,c2,c3,c4 = st.columns([1,1,1,1])
    nome_default = st.session_state.get(f"nome_{i}", default_names[i] if i < len(default_names) else f"Var{i+1}")
    nome = c1.text_input(f"nome_{i}", value=nome_default, label_visibility="collapsed", key=f"nome_{i}")
    vmin = c2.number_input(f"vmin_{i}", value=0.0, step=1.0, label_visibility="collapsed", format="%.6f", key=f"vmin_{i}")
    vmax = c3.number_input(f"vmax_{i}", value=1000.0, step=1.0, label_visibility="collapsed", format="%.6f", key=f"vmax_{i}")
    k = c4.number_input(f"k_{i}", min_value=1, value=2, step=1, label_visibility="collapsed", key=f"k_{i}")
    var_defs.append({"nome": nome.strip(), "vmin": float(vmin), "vmax": float(vmax), "k": int(k)})

expr_default = st.session_state.get("expr_text", "(2624-PE*3280)/5 + (884-Eixo*1105)/7 + (3870-4837*PI)/7 + 5700")
expr_input = st.text_area("Expressão", value=expr_default, height=120, key="expr_text")
expr = parse_expressao(expr_input)

nome_saida = st.text_input("Nome da variável de saída", value="RN")
gerar = st.button("🚀 Gerar tabela")

if gerar:
    try:
        fator = st.session_state.get("fator_val", 0.05)
        niveis_por_var = {}
        vmax_rd = {}
        for v in var_defs:
            niveis, _, vrd = gerar_niveis_por_variavel(v["k"], v["vmin"], v["vmax"], fator)
            niveis_por_var[v["nome"]] = niveis
            vmax_rd[v["nome"]] = vrd

        ordered_vars = ordenar_variaveis_por_maximo(var_defs, expr, niveis_por_var, vmax_rd)
        linhas_vals, ks, blocks = construir_tabela_linear(ordered_vars, niveis_por_var)

        html = gerar_html(ordered_vars, ks, blocks, niveis_por_var, vmax_rd, linhas_vals, expr, nome_saida, fator)
        csv_bytes = gerar_csv_bytes(ordered_vars, niveis_por_var, vmax_rd, linhas_vals, expr, nome_saida, fator)

        st.components.v1.html(html, height=600, scrolling=True)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Baixar CSV", data=csv_bytes, file_name="faixas.csv", mime="text/csv")
        c2.download_button("⬇️ Baixar HTML", data=html.encode("utf-8"), file_name="faixas.html", mime="text/html")

    except Exception as e:
        st.error(f"Erro: {e}")

# -------------------- Execução automática --------------------
if __name__ == "__main__":
    import os, sys, threading, time, webbrowser
    from streamlit.web import cli as stcli

    def _in_streamlit_runtime() -> bool:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            return get_script_run_ctx() is not None
        except Exception:
            return False

    if not _in_streamlit_runtime():
        # Evita duplicar abas
        os.environ["BROWSER"] = "none"
        port = 8501
        url = f"http://localhost:{port}"

        # Função para abrir o navegador 1 segundo depois
        def abrir_browser():
            time.sleep(1.5)
            webbrowser.open_new_tab(url)

        threading.Thread(target=abrir_browser, daemon=True).start()

        # Mostra mensagem caso execute em modo terminal
        print(f"🌐 Servidor iniciado — acesse: {url}")

        sys.argv = [
            "streamlit",
            "run",
            __file__,
            f"--server.port={port}",
            "--server.headless=false"
        ]
        sys.exit(stcli.main())
