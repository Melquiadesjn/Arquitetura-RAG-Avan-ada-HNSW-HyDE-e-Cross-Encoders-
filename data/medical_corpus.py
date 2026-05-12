"""
Corpus simulado de fragmentos de manuais médicos técnicos.
Gerado com auxílio de IA e revisado para garantir jargão clínico adequado.
"""

MEDICAL_DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "doc_001",
        "title": "Cefaleia Pulsátil e Fotofobia — Diagnóstico Diferencial",
        "text": (
            "A cefaleia pulsátil de forte intensidade associada à fotofobia e fonofobia "
            "constitui quadro clássico de enxaqueca com aura (migrânea com aura, CID-10 G43.1). "
            "O diagnóstico diferencial deve incluir hipertensão intracraniana idiopática, "
            "meningite asséptica e hemorragia subaracnóidea. A escala de dor VAS ≥ 7 combinada "
            "ao sinal de Kernig positivo indica punção lombar emergencial para afastar xantocromia."
        ),
    },
    {
        "id": "doc_002",
        "title": "Protocolo de Manejo da Dor Torácica Aguda",
        "text": (
            "Síndrome coronariana aguda (SCA) apresenta dor precordial opressiva irradiada "
            "para membro superior esquerdo, mandíbula e dorso. Biomarcadores cardíacos: "
            "troponina I ultrassensível > 52 ng/L (99º percentil) confirma necrose miocárdica. "
            "ECG de 12 derivações deve ser obtido em ≤ 10 minutos. Supradesnivelamento de ST "
            "≥ 1 mm em duas derivações contíguas define IAMCSST, requerendo reperfusão imediata "
            "por ICP primária ou trombólise farmacológica conforme protocolo TIMI."
        ),
    },
    {
        "id": "doc_003",
        "title": "Avaliação Neurológica da Perda de Consciência Transitória",
        "text": (
            "Síncope vasovagal é a etiologia mais prevalente de perda de consciência transitória "
            "(PCT), correspondendo a 21% dos casos em pronto-socorro. Diagnóstico clínico baseia-se "
            "no tilt-test (sensibilidade 70-80%) e na monitorização Holter de 24h para excluir "
            "arritmias paroxísticas. Síncope de origem cardíaca (bloqueio AV completo, taquicardia "
            "ventricular sustentada) apresenta risco de morte súbita e requer implante de CDI "
            "conforme escore EGSYS ≥ 3."
        ),
    },
    {
        "id": "doc_004",
        "title": "Insuficiência Respiratória Aguda — Classificação e Suporte Ventilatório",
        "text": (
            "Insuficiência respiratória hipoxêmica (tipo I): PaO₂ < 60 mmHg com FiO₂ ambiente. "
            "SDRA (síndrome do desconforto respiratório agudo) definida por: início agudo ≤ 7 dias, "
            "opacidades bilaterais na TC, razão PaO₂/FiO₂ ≤ 300 mmHg, ausência de hipertensão "
            "atrial esquerda. Estratégia protetora: volume corrente 6 mL/kg de peso predito, "
            "PEEP titulado pela tabela ARDSNet, pressão de platô ≤ 30 cmH₂O. Posição prona "
            "≥ 16h/dia reduz mortalidade em SDRA grave (PaO₂/FiO₂ < 150)."
        ),
    },
    {
        "id": "doc_005",
        "title": "Diabetes Mellitus Tipo 2 — Critérios Diagnósticos e Metas Glicêmicas",
        "text": (
            "Critérios ADA 2024 para DM2: glicemia de jejum ≥ 126 mg/dL em duas ocasiões, "
            "HbA1c ≥ 6,5%, glicemia pós-prandial 2h ≥ 200 mg/dL no TOTG, ou glicemia aleatória "
            "≥ 200 mg/dL com sintomas. Meta HbA1c < 7% para maioria dos pacientes; < 8% em idosos "
            "frágeis. Metformina permanece primeira linha (grau A de evidência). Inibidores SGLT-2 "
            "e agonistas GLP-1 indicados quando DCV estabelecida, DRC estágio ≥ 3 ou IC."
        ),
    },
    {
        "id": "doc_006",
        "title": "Sepse e Choque Séptico — Bundle de 1 e 3 Horas",
        "text": (
            "Sepse: disfunção orgânica ameaçadora à vida causada por resposta desregulada do "
            "hospedeiro à infecção (Sepsis-3). SOFA score ≥ 2 pontos confirma disfunção. "
            "Bundle de 1 hora: hemoculturas (×2 sítios), lactato sérico, antibioticoterapia "
            "de amplo espectro, 30 mL/kg de cristaloide se hipotensão ou lactato ≥ 4 mmol/L. "
            "Choque séptico: vasopressor necessário para PAM ≥ 65 mmHg após ressuscitação volêmica "
            "e lactato > 2 mmol/L. Norepinefrina é vasopressor de primeira escolha."
        ),
    },
    {
        "id": "doc_007",
        "title": "Hipotireoidismo Primário — Diagnóstico e Reposição Hormonal",
        "text": (
            "Hipotireoidismo primário: TSH > 4,5 mUI/L com T4 livre < 0,8 ng/dL. Sintomas: "
            "fadiga, ganho ponderal, intolerância ao frio, constipação, mixedema, bradicardia "
            "sinusal. Anticorpos anti-TPO positivos confirmam tireoidite de Hashimoto (causa "
            "autoimune mais comum). Reposição: levotiroxina sódica 1,6 µg/kg/dia em jejum, "
            "ajuste pela meta de TSH 1-2,5 mUI/L. Gravidez: TSH < 2,5 mUI/L no 1º trimestre."
        ),
    },
    {
        "id": "doc_008",
        "title": "Acidente Vascular Cerebral Isquêmico — Janela Terapêutica e Trombólise",
        "text": (
            "AVC isquêmico agudo: oclusão arterial cerebral causa infarto em minutos. "
            "Escala NIHSS quantifica déficit neurológico (0-42). Tomografia sem contraste "
            "exclui hemorragia antes de trombolítico. Alteplase IV (0,9 mg/kg, máx 90 mg): "
            "janela de 4,5 horas desde início dos sintomas (evidência Classe I, Nível A). "
            "Trombectomia mecânica indicada até 24h em oclusão de grande vaso (ASPECTS ≥ 6). "
            "Anticoagulação plena contraindicada nas primeiras 24h pós-trombólise."
        ),
    },
    {
        "id": "doc_009",
        "title": "Hipertensão Arterial Sistêmica — Classificação e Estratégias Anti-hipertensivas",
        "text": (
            "HAS: PAS ≥ 140 mmHg e/ou PAD ≥ 90 mmHg em duas aferições distintas. Estágio 1: "
            "140-159/90-99; Estágio 2: 160-179/100-109; Estágio 3: ≥ 180/110. Lesão de órgão-alvo: "
            "HVE, retinopatia, microalbuminúria, placa aterosclerótica carotídea. Fármacos de "
            "primeira linha: IECA/BRA (nefroprotegidos), BCC di-hidropiridínico, tiazídico. "
            "Associação IECA + BRA contraindicada (risco de hipercalemia e IRA)."
        ),
    },
    {
        "id": "doc_010",
        "title": "Anemia Ferropriva — Investigação e Reposição de Ferro",
        "text": (
            "Anemia ferropriva: hemoglobina < 12 g/dL (mulheres) ou < 13 g/dL (homens) com "
            "ferritina sérica < 30 ng/mL e saturação de transferrina < 16%. VCM < 80 fL "
            "(microcitose), CHCM < 32 g/dL (hipocromia). Investigação obrigatória da causa: "
            "endoscopia digestiva em homens > 50 anos e mulheres pós-menopausa para excluir "
            "neoplasia colorretal. Reposição: sulfato ferroso 200 mg 3×/dia por 3-6 meses; "
            "ferro sacarose IV quando intolerância oral ou má absorção (doença de Crohn)."
        ),
    },
    {
        "id": "doc_011",
        "title": "Insuficiência Cardíaca com Fração de Ejeção Reduzida (ICFEr)",
        "text": (
            "ICFEr: FEVE < 40% ao ecocardiograma. Quadro clínico: dispneia paroxística noturna, "
            "ortopneia, edema bilateral de membros inferiores, terceira bulha (B3), crepitações "
            "pulmonares. BNP > 400 pg/mL confirma descompensação. Tratamento modificador: "
            "IECA (ou sacubitril/valsartana), betabloqueador (carvedilol/bisoprolol), "
            "antagonista mineralocorticoide, inibidor SGLT-2 (dapagliflozina/empagliflozina). "
            "Quadrupla terapia reduz mortalidade cardiovascular em 50%."
        ),
    },
    {
        "id": "doc_012",
        "title": "Pneumonia Adquirida na Comunidade — Escore PSI e Antibioticoterapia",
        "text": (
            "PAC: consolidação pulmonar de origem infecciosa adquirida fora do ambiente hospitalar. "
            "Escore PSI/PORT estratifica risco: Classe I-II (ambulatorial), III (observação), "
            "IV-V (internação/UTI). Critérios CURB-65: confusão mental, ureia > 43 mg/dL, FR ≥ 30, "
            "PAS < 90 ou PAD ≤ 60, idade ≥ 65. Agente mais comum: Streptococcus pneumoniae. "
            "Antibiótico: amoxicilina + clavulanato + macrolídeo (PAC leve); "
            "betalactâmico + macrolídeo ou fluoroquinolona respiratória (PAC grave)."
        ),
    },
    {
        "id": "doc_013",
        "title": "Doença Renal Crônica — Estadiamento e Nefroproteção",
        "text": (
            "DRC: TFGe < 60 mL/min/1,73m² ou marcadores de lesão renal por > 3 meses. "
            "Estadiamento KDIGO por TFGe: G1(≥90), G2(60-89), G3a(45-59), G3b(30-44), "
            "G4(15-29), G5(<15). Albuminúria: A1(<30), A2(30-300), A3(>300 mg/g creatinina). "
            "Nefroproteção: IECA/BRA em proteinúria, controle pressórico < 130/80, "
            "HbA1c < 7%, inibidor SGLT-2 (fanerenona + dapagliflozina na DRC diabética). "
            "Evitar AINEs e contrastes iodados em G3-G5."
        ),
    },
    {
        "id": "doc_014",
        "title": "Fibrilação Atrial — Estratificação de Risco Tromboembólico e Anticoagulação",
        "text": (
            "FA: arritmia sustentada mais prevalente (2-4% da população adulta). Classificação: "
            "paroxística (< 7 dias), persistente (> 7 dias), permanente. Risco tromboembólico: "
            "CHA₂DS₂-VASc ≥ 2 (homens) ou ≥ 3 (mulheres) indica anticoagulação oral. "
            "ACO de escolha: DOACs (apixabana, rivaroxabana, dabigatrana) superiores à varfarina "
            "em segurança intracraniana. Risco hemorrágico: HAS-BLED ≥ 3 alerta, mas não "
            "contraindica anticoagulação — corrigir fatores modificáveis."
        ),
    },
    {
        "id": "doc_015",
        "title": "Epilepsia — Classificação das Crises e Antiepilépticos de Primeira Linha",
        "text": (
            "Epilepsia: ≥ 2 crises não provocadas com intervalo > 24h, ou uma crise com risco "
            "de recorrência ≥ 60%. Classificação ILAE 2017: focal (consciência preservada ou "
            "comprometida), generalizada (ausência, mioclônica, tônico-clônica), desconhecida. "
            "EEG e RNM cerebral obrigatórios na investigação inicial. Antiepilépticos de 1ª "
            "linha: ácido valpróico (generalizada), lamotrigina/levetiracetam (focal). "
            "Estado epiléptico: benzodiazepínico IV até 20 min; fenitoína ou valproato IV após."
        ),
    },
    {
        "id": "doc_016",
        "title": "Osteoporose — Densitometria e Prevenção de Fraturas",
        "text": (
            "Osteoporose: T-score ≤ -2,5 na densitometria óssea (DXA) em colo femoral ou "
            "coluna lombar (L1-L4). Osteopenia: T-score -1,0 a -2,5. Risco de fratura em "
            "10 anos: FRAX® considera idade, gênero, DMO, corticoterapia crônica, tabagismo, "
            "alcoolismo, artrite reumatoide. Tratamento farmacológico: bisfosfonatos "
            "(alendronato 70 mg/semana) de primeira escolha; denosumabe 60 mg SC a cada "
            "6 meses em intolerância; romosozumabe em osteoporose grave com fratura prévia."
        ),
    },
    {
        "id": "doc_017",
        "title": "Choque Hipovolêmico — Classificação e Ressuscitação Volêmica",
        "text": (
            "Choque hipovolêmico hemorrágico: déficit de volume intravascular por perda "
            "sanguínea. Classificação ATLS: Classe I (< 15% volemia, FC < 100), Classe II "
            "(15-30%, FC 100-120), Classe III (30-40%, FC 120-140, hipotensão), Classe IV "
            "(> 40%, FC > 140, anúria). Ressuscitação balanceada: hemácias:plasma:plaquetas "
            "1:1:1. Ácido tranexâmico 1g IV nas primeiras 3h pós-trauma. Evitar "
            "hiper-ressuscitação com cristaloides (acidose hiperclorêmica, coagulopatia dilucional)."
        ),
    },
    {
        "id": "doc_018",
        "title": "Doença Inflamatória Intestinal — Colite Ulcerativa vs. Doença de Crohn",
        "text": (
            "DII engloba colite ulcerativa (CU) e doença de Crohn (DC). CU: inflamação "
            "mucosa contínua do reto (proctite) até cólon total; sem granulomas na biópsia. "
            "DC: inflamação transmural segmentar, qualquer segmento do TGI; granulomas não "
            "caseosos patognomônicos. Biomarcadores: calprotectina fecal > 250 µg/g e "
            "PCR elevada indicam atividade. Indução: corticosteroide sistêmico; "
            "manutenção: mesalazina (CU), azatioprina ou biológico anti-TNF (infliximabe) "
            "em doença moderada-grave."
        ),
    },
    {
        "id": "doc_019",
        "title": "Hiperkalemia Grave — Manejo de Emergência",
        "text": (
            "Hiperkalemia: K⁺ sérico > 5,5 mEq/L. Grave (> 6,5 mEq/L) ou com alterações "
            "eletrocardiográficas (onda T apiculada, QRS alargado, onda sinusoidal). "
            "Sequência de tratamento: (1) Gluconato de cálcio 10% 10 mL IV em 2-3 min — "
            "estabilização miocárdica imediata; (2) Insulina regular 10 UI + glicose 50% 50 mL — "
            "deslocamento intracelular em 15-30 min; (3) Bicarbonato de sódio se acidose "
            "metabólica; (4) Patiromer/SZC — eliminação intestinal; (5) Hemodiálise em "
            "refratariedade ou anúria."
        ),
    },
    {
        "id": "doc_020",
        "title": "Anafilaxia — Diagnóstico e Tratamento Imediato",
        "text": (
            "Anafilaxia: reação de hipersensibilidade sistêmica grave e potencialmente fatal. "
            "Critérios diagnósticos: início agudo com envolvimento cutâneo + comprometimento "
            "respiratório ou cardiovascular, ou dois ou mais sistemas após exposição ao alérgeno. "
            "Tratamento de primeira linha: adrenalina 0,3-0,5 mg IM na face anterolateral da "
            "coxa (não subcutânea). Posição de Trendelenburg se hipotensão. Anti-histamínico "
            "e corticosteroide são adjuvantes, nunca substitutos da adrenalina. "
            "Observação mínima de 4-6h por risco de reação bifásica."
        ),
    },
    {
        "id": "doc_021",
        "title": "Insônia Crônica — Critérios Diagnósticos e Terapia Cognitivo-Comportamental",
        "text": (
            "Insônia crônica: dificuldade de iniciar ou manter o sono ≥ 3 noites/semana por "
            "≥ 3 meses, com comprometimento diurno (fadiga, déficit cognitivo, humor). "
            "Polissonografia não é obrigatória no diagnóstico primário. TCC-I (Terapia "
            "Cognitivo-Comportamental para Insônia) é tratamento de primeira linha (superior "
            "a farmacoterapia a longo prazo). Componentes: restrição do sono, controle de "
            "estímulos, higiene do sono, reestruturação cognitiva. Farmacológico: zolpidem "
            "ou lemborexant em curto prazo; evitar benzodiazepínicos crônicos (risco de "
            "dependência e quedas em idosos)."
        ),
    },
    {
        "id": "doc_022",
        "title": "Glaucoma de Ângulo Aberto — Monitorização da Pressão Intraocular",
        "text": (
            "Glaucoma primário de ângulo aberto (GPAA): neuropatia óptica progressiva com "
            "perda de campo visual e PIO geralmente > 21 mmHg (tonometria de aplanação de "
            "Goldmann). Disco óptico: escavação (cup/disc ratio) > 0,6, hemorragia de bordo "
            "e assimetria entre olhos. Perimetria computadorizada (campo visual 24-2) detecta "
            "defeitos feixiculares. Tratamento tópico inicial: análogos de prostaglandina "
            "(latanoprosta 0,005%). Trabeculoplastia a laser (SLT) como alternativa. "
            "Cirurgia filtrante (trabeculectomia) em PIO incontrolável."
        ),
    },
    {
        "id": "doc_023",
        "title": "Pancreatite Aguda — Escore de Gravidade e Manejo",
        "text": (
            "Pancreatite aguda: ativação prematura do tripsinogênio dentro do pâncreas causa "
            "autodigestão. Causas: litíase biliar (40%), etanol (30%). Diagnóstico: dor "
            "epigástrica em faixa + amilase ou lipase > 3× o limite superior do normal. "
            "Escore BISAP ≥ 3 ou APACHE-II ≥ 8 prediz pancreatite grave. TC abdome com "
            "contraste (Balthazar E/CTSI ≥ 7) identifica necrose. Tratamento: hidratação "
            "agressiva (Ringer-Lactato 250-500 mL/h nas primeiras 24h), analgesia, jejum "
            "oral apenas se vômitos incoercíveis; nutrição enteral precoce preferida."
        ),
    },
    {
        "id": "doc_024",
        "title": "Distúrbios Ácido-Base — Interpretação Sistemática da Gasometria Arterial",
        "text": (
            "Interpretação gasométrica: (1) pH — < 7,35 acidemia, > 7,45 alcalemia. "
            "(2) PaCO₂ — < 35 hiperventilação, > 45 hipoventilação. "
            "(3) HCO₃⁻ — < 22 acidose metabólica, > 26 alcalose metabólica. "
            "Compensação esperada: acidose metabólica → PaCO₂ = 1,5×HCO₃⁻ + 8 ± 2 (Fórmula de "
            "Winter). Ânion gap = Na⁺ - (Cl⁻ + HCO₃⁻); normal 8-12 mEq/L. "
            "AG elevado: MUDPILES (metanol, uremia, diabética, propileno glicol, isoniazida, "
            "lactato, etilenoglicol, salicilatos)."
        ),
    },
    {
        "id": "doc_025",
        "title": "Crise Hipertensiva — Emergência vs. Urgência Hipertensiva",
        "text": (
            "Emergência hipertensiva: PA gravemente elevada (geralmente > 180/120 mmHg) com "
            "lesão aguda de órgão-alvo (encefalopatia, AVC hemorrágico, dissecção aórtica, "
            "EAP, SCA, eclâmpsia). Redução de PA em 25% na 1ª hora com anti-hipertensivo IV "
            "(nicardipina, labetalol, nitroprussiato). Exceção: AVC isquêmico agudo — não "
            "tratar < 220/120 mmHg. Urgência hipertensiva: PA elevada sem lesão aguda; "
            "redução gradual em 24-48h com via oral (captopril, clonidina). "
            "Não há benefício em redução abrupta na urgência."
        ),
    },
]
