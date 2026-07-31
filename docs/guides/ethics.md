# Ética e LGPD

!!! danger "Leia antes de coletar qualquer dado"
    Este projeto trata **dados pessoais sensíveis de saúde** de pessoas
    identificáveis na origem. A coleta sem aprovação prévia de Comitê de Ética
    em Pesquisa é violação de protocolo — e o pipeline a bloqueia por barreira
    técnica.

## Recursos de apoio

Se você está passando por sofrimento psíquico, ou conhece alguém nessa situação:

| Serviço | Contato |
|---|---|
| **CVV** — Centro de Valorização da Vida | **188** (24 h, gratuito) · [cvv.org.br](https://www.cvv.org.br) |
| **CAPS** — Centros de Atenção Psicossocial | Unidades do SUS em todo o país |
| **SAMU** | **192** |

---

## Barreira ética no código

A etapa `collect` verifica `ETHICS_APPROVAL_ID` no `.env` antes de qualquer
requisição:

```
EthicalGateError: Coleta bloqueada: nenhuma aprovação CEP/CONEP registrada.
Defina ETHICS_APPROVAL_ID no .env com o número do CAAE.
```

Existe uma saída para teste técnico com contas próprias:

```bash
python src/main.py --stage collect --allow-collection-without-ethics
```

O uso é registrado no log com aviso explícito. **Dados coletados assim não podem
compor a base da pesquisa.**

## Submissão ao CEP/CONEP

O protocolo é submetido pela [Plataforma Brasil](https://plataformabrasil.saude.gov.br)
e deve descrever, no mínimo:

- **Objetivo e desenho** — estudo observacional, sem intervenção.
- **Fonte dos dados** — publicações públicas do X/Twitter.
- **Dispensa do TCLE** — justificada pelo caráter público dos dados, pelo
  volume que inviabiliza consentimento individual e pelas salvaguardas de
  anonimização. **A dispensa é decisão do comitê, não do pesquisador.**
- **Riscos** — reidentificação, estigmatização, uso indevido do modelo.
- **Salvaguardas** — as descritas abaixo.
- **Retenção e descarte** — prazo de guarda e eliminação ao término.
- **Devolutiva** — como os resultados retornam à sociedade.

## Base legal (LGPD)

O tratamento se apoia em:

- **Art. 7º, IV** — realização de estudos por órgão de pesquisa, com
  anonimização sempre que possível.
- **Art. 11, II, "c"** — tratamento de dado sensível para estudos por órgão de
  pesquisa.
- **Art. 4º, II, "b"** — o tratamento para fins exclusivamente acadêmicos tem
  regime próprio.

O enquadramento deve ser confirmado com a assessoria jurídica da instituição.

## Salvaguardas implementadas

### Pseudonimização na ingestão

Identificadores diretos viram SHA-256 com salt **antes** de qualquer gravação em
disco. Nenhum handle chega a ser persistido.

```python
def pseudonymize(identifier: str | int, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}{identifier}".encode()).hexdigest()
    return f"u_{digest[:16]}"
```

O salt é obrigatório e vem do `.env`. Sem ele, o hash seria reversível por força
bruta: o espaço de handles é público e enumerável.

### Minimização de dados

Não são coletados: nome de exibição, biografia, foto, localização declarada, URL
de perfil, **nem qualquer atributo demográfico**.

O que se coleta: texto das publicações, carimbo temporal, contagens de
engajamento e contagens de audiência.

### Remoção de PII do texto

Menções, URLs, e-mails e telefones são substituídos por placeholders na
normalização.

### Redação de PII nos logs

O filtro é anexado a **todos** os handlers de logging. Depender da disciplina de
quem escreve `logger.info` não seria garantia, e um log com PII de pessoas em
sofrimento psíquico é exatamente o vazamento que o projeto não pode ter.

### Processamento local do LLM

A extração de atributos psicológicos usa Ollama na própria máquina. Enviar os
textos a uma API remota configuraria transferência internacional de dado pessoal
sensível sem base legal adequada (arts. 11 e 33).

### Não redistribuição

O dataset **não é publicado**, mesmo pseudonimizado: a reidentificação por busca
do texto exato é trivial numa rede social pública.

O que é publicado: o código, os termos das consultas, os léxicos, as
configurações e os resultados agregados. Terceiros com aprovação ética própria
podem reconstruir uma base equivalente.

### Direito à eliminação

Cada usuário tem um arquivo próprio em
`data/raw/user_histories/<user_id>.parquet`. A remoção de um titular é uma
operação localizada, seguida da reexecução do pipeline.

---

## Limitações declaradas

Estas limitações estão no model card e no datasheet — não são rodapé.

### Viés de seleção na classe controle

A classe controle é coletada por temas neutros (`#python`, `#futebol`, ...).
Postar sobre futebol não garante ausência de sofrimento psíquico.

**O rótulo `controle` significa "sem sinais *detectados* no recorte coletado" —
nunca "ausência clínica confirmada".** Qualquer afirmação de "especificidade" do
modelo precisa ser lida sob essa restrição.

### Rótulos sem validação clínica

Os rótulos vêm de supervisão fraca combinada com revisão manual por não
especialistas. Nenhum foi confirmado por profissional de saúde mental nem por
instrumento psicométrico validado.

A concordância com a revisão manual (kappa de Cohen) delimita o teto de
desempenho: **nenhum modelo pode superar consistentemente a qualidade do rótulo
com que foi treinado.**

### Ausência de auditoria de fairness demográfica

O projeto não coleta sexo, idade, raça ou região, por minimização de dados. Uma
auditoria de justiça demográfica exigiria coletar exatamente a informação
sensível que se optou por não coletar.

A alternativa adotada é a avaliação por fatias **comportamentais** (volume de
publicação, janela de observação, atividade noturna). É a auditoria que os dados
disponíveis permitem fazer honestamente, e a limitação está declarada.

### Viés de plataforma e de vocabulário

Usuários do X/Twitter que escrevem publicamente sobre sofrimento não representam
a população brasileira. Além disso, os termos das consultas determinam quem entra
na amostra: formas de expressar sofrimento fora desse vocabulário — por região,
faixa etária ou grupo social — ficam invisíveis ao estudo.

---

## Usos fora de escopo

O modelo **não** deve ser usado para:

- :material-close: **Diagnóstico clínico.** Não diagnostica nada.
- :material-close: **Decisão automatizada sobre pessoas.** Qualquer uso
  operacional exige revisão humana.
- :material-close: **Vigilância ou triagem sem consentimento** de indivíduos
  identificáveis.
- :material-close: **Decisões de emprego, seguro ou crédito.**
- :material-close: **Generalização** para outros idiomas, plataformas ou
  populações não observadas.

### Risco de uso indevido

Um modelo que sinaliza risco de suicídio pode ser usado para vigilância ou
estigmatização — o mesmo sinal que permite oferecer ajuda permite discriminar.

Por isso: escopo restrito à pesquisa, dados não redistribuídos, usos fora de
escopo declarados na licença e no model card.

---

## Comunicação responsável

Ao divulgar resultados, seguir as
[diretrizes da OMS para a cobertura do suicídio](https://www.who.int/publications/i/item/9789240057555):

- **Não** publicar exemplos reais de tweets, mesmo anonimizados.
- **Não** apresentar métricas como se fossem acurácia diagnóstica.
- **Sempre** incluir os canais de apoio (CVV 188).
- **Sempre** declarar as limitações junto dos resultados, não em nota de rodapé.
- Evitar linguagem sensacionalista ("IA que detecta suicídio").
