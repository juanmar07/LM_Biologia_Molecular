"""Dataset v2 de fuentes originales: afirmaciones atómicas + pares de evidencia.

--draft genera una propuesta mutable; --freeze audita y bloquea los artefactos finales.
No hay expansión por paráfrasis: una pregunta por afirmación y duplicación intencional
solo en los dos miembros de un par contrastivo (contextos distintos, respuesta opuesta).
"""
import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from statistics import mean

from dataset_sft_5A1 import (BASE, DATA, RESULTS, ABSTENTIONS, TASKS, documents, require,
    write_json, sha256, encode, prompt, hashes, verify_preservation, log_accesses)
from tokenizador import TokenizadorEducativo

VERSION = '5A.1-v2.0'
SEED = 42
TOPICS = {
    'epigenética': r'epigen|metil|histon|acetil|cromatina|impronta',
    'células_madre': r'células madre|totipot|pluripot|multipot|oligopot|unipot|blastocisto|autorregenera|diferenciación',
    'terapia_génica': r'terapia génica|terapéutic|transgén|adenoviral|lentivir|adenovirus|retrovir',
    'ciencias_ómicas': r'\b(?:ómicas?|ómicos?|genómica|genómico|genómicas|genómicos|proteoma|transcriptoma|bioinformática)\b',
    'clonación_vectores': r'clonación|plásmid|cósmid|vector|clonado|fago|bacteriófago',
    'replicación': r'replic|polimerasa|hebra|horquilla|ligasa|cebador|ori\b',
    'transcripción_regulación': r'transcrip|promotor|operón|expresión génica|expresión de gen|ARN pol|RNA pol',
    'traducción': r'traducci|ribosoma|polipéptid|aminoácid|código genético',
    'ingeniería_genética': r'CRISPR|Cas9|restricción|recombin|PCR|secuenciación|electroforesis|hibridación',
    'genética_molecular': r'ADN|DNA|ARN|RNA|gen|cromosom|nucleótido|mutación|alelo|proteína|enzima|célula|mitosis|meiosis|telómero|nucleósido',
}
SCIENCE = re.compile('|'.join('(?:'+p+')' for p in TOPICS.values()), re.I)
VERBS = [
    'se define como','se definen como','se denomina','se denominan','se conoce como','se conocen como',
    'se caracteriza por','se caracterizan por','se clasifica','se clasifican','se convierte','se convierten',
    'se divide','se dividen','se une','se unen','se sintetiza','se sintetizan','se encuentra','se encuentran',
    'se replica','se replican','se transmite','se transmiten','se utiliza','se utilizan','se realiza','se realizan',
    'se incorpora','se incorporan','se separa','se separan','se expresa','se expresan','se obtiene','se obtienen',
    'se inicia','se inician','se localiza','se localizan','se mantiene','se mantienen','se desarrolla','se desarrollan',
    'se basa','se basan','se asocia','se asocian','se aísla','se aíslan','se forma','se forman',
    'se agrupa','se agrupan','se produce','se producen','se regula','se regulan','se determina','se determinan',
    'se degrada','se degradan','se añade','se añaden','se elimina','se eliminan','se modifica','se modifican',
    'está constituido por','están constituidos por','está constituida por','están constituidas por',
    'está formado por','están formados por','está formada por','están formadas por',
    'ha permitido','han permitido','ha demostrado','ha acelerado','ha abierto','ha ampliado','fueron obtenidas','fue desarrollada',
    'está','están','es','son','permite','permiten','presenta','presentan','contiene','contienen',
    'produce','producen','forma','forman','participa','participan','regula','regulan','origina','originan',
    'genera','generan','depende','dependen','puede','pueden','tiene','tienen','cambia','cambian',
    'codifica','codifican','actúa','actúan','consiste','consisten','reconoce','reconocen','representa','representan',
    'incluye','incluyen','protege','protegen','requiere','requieren','evita','evitan','induce','inducen',
    'inhibe','inhiben','cataliza','catalizan','resulta','resultan','interviene','intervienen',
    'corresponde','corresponden','consta','constan','posee','poseen','mantiene','mantienen',
    'aumenta','aumentan','disminuye','disminuyen','pertenece','pertenecen','desempeña','desempeñan',
    'necesita','necesitan','permanece','permanecen','transporta','transportan','inicia','inician',
    'sintetiza','sintetizan','determina','determinan','dirige','dirigen','aporta','aportan',
    'funciona','funcionan','sirve','sirven','comprende','comprenden','ocurre','ocurren','incrementa','incrementan',
    'constituye','constituyen','refiere','conduce','conllevan','conlleva','controla','controlan','provoca','provocan',
    'sufre','sufren','progresa','progresan','destruye','destruyen',
]
RELATION = re.compile(r'^((?:El|La|Los|Las|Un|Una|ADN|DNA|ARN|RNA) .{1,95}?) ((?:(?:no|también|generalmente|normalmente|siempre|a menudo) )?(?:' + '|'.join(re.escape(v) for v in sorted(VERBS,key=len,reverse=True)) + r'))\b[ ,]+(.+)\.$', re.I)
BAD_SUBJECT = re.compile(r'\b(?:que|cual|se|no|es|son|está|están|estará|estarán|sufren|hacen|hace|sugieren|vez|primeros|segundos|muestra|individuo|paciente|ya|suelen|si|sólo|solo|entonces|contiguo|referencia|programa|las posiciones|cuatro calles)\b|[,;:]|\b(?:y|o|de|del|en|por|con|a|d)\s*$', re.I)
DEPENDENT = re.compile(r'\b(?:figura|cuadro|tabla|diapositiva|capítulo|pág|anterior|anteriormente|siguiente|éste|ésta|éstos|éstas|este|esta|estos|estas|ese|esa|esos|esas|ambos|ambas|dicho|dicha|ello|aquí|autor|copyright|doi|et al|nosotros|mencionada|mencionado|comentado|colaboradores|cols)\b|https?://|[�•♦]|[a-záéíóú]-\s+[a-záéíóú]', re.I)
NOISE = re.compile(r'\b(?:fi gura|efi ciente|infl uir|transcripciona l|acualquier|producir se|induc iendo|codifi cadora|específi ca|identifi cación|polimorfi smos|pres enta|vent ajas|muyrelacionados|especiesdiferentes|induciblegen|inductorinductor|activadoractivador|ARNmensajero|dos modificaciones|varios mecanismos|distintos mecanismos|muy variados|fi lamentos|modifi cada|modifi caciones|fl uidos|superfi cie|hemofi lia|constitu ye|aminoá cidos)\b|\d\.$|\b(?:El|La|Los|Las) .+ \b(?:El|La|Los|Las) [A-Z]{2,}\b')
AMBIGUOUS = re.compile(r'^(?:la proteína|el fragmento de ADN|las tres enzimas|una sola mutación|el ARN viral|el gen responsable.*|la enzima más empleada|la preparación de partículas víricas.*)$', re.I)
STOP = set('el la los las un una unos unas de del a al en por con y o que se su sus es son para como entre sobre también normalmente siempre gen genes célula células proteína proteínas enzima enzimas ADN DNA ARN RNA'.lower().split())

# Revisión de las afirmaciones candidatas contra sus propios fragmentos. No se
# corrige la ciencia de una fuente: se excluyen referentes ausentes, enumeraciones
# incompletas y texto roto. Las reglas se aplican también a contextos negativos.
REVIEW_NOISE = re.compile(r'\b\w*fi \w+|\b\w*fl \w+|\b(?:pre - sencia|Secuencia-ción|bloquearsu|línealinfoide|120células|diver-sidad|cromatin a|comoen|unmicroambiente|subexp resión|des-tacada|80a|deltrasplante|deprácticamente|célulasdel|hematopoyéticas,cardiomiocitos|enfermedad es|acélulas|derechooperador|lapulpa|cultivoespecífico|adultascélulas|devarios|trasplantede|otrasproteínas|mismopotencial|cistróncistrón|sombreadasombreada|proteínaMax|represorrepresor|laclac|enzimafotoliasa|cróni-codegenerativas|fo rma|vari aciones)\b',re.I)
REVIEW_SUBJECT = re.compile(r'^(?:el objetivo final|un ejemplo|la concentración de la enzima|el resultado|la causa molecular de la enfermedad|la búsqueda|la reacción|la molécula entera|los más utilizados|el cuarto factor|el laboratorio|el factor transcripcional|el ARNm remanente|las células resultantes|las proteínas así separadas|la infección|el ADN|la integración|el proceso|una ventaja adicional|las otras subunidades|el tercer punto|una forma indirecta pero muy rápida|las pocas diferencias existentes|la proteína expresada|el objetivo|las partículas de virus recién generadas|las excepciones|las consecuencias fenotípicas|el protocolo tradicional básico|el caso más frecuente|la mezcla inicial|los efectos más notorios|la estabilidad|la forma mutada del gen oncosupresor|las otras dos subunidades|la o las moléculas de rDNA plasmídico incorporadas|el principal problema|el método más utilizado actualmente|un concepto similar pero diferente|los criterios seguidos en general|el virus|una de ellas|los genes|la región a amplificar|una proteína|una enfermedad|las letras FT marcan otras características de la secuencia como|la opción de producir árboles condensados|el vector de clonación|la región repetida)$',re.I)
REVIEW_CLAIM = re.compile(r'por una parte, el ADNc\.$|dos células: la primera|genes presentes de forma exclusiva en mamíferos|por el contrario|\by\.$|\bes decir\.$|dos razones: a\.$|identificar genes que controlan\.$|ya fue explicado|\b(?:ver el capitulo|izquierda|derecha)\b|\(AA\)|\(99\)|\(sombread|ordenados por el estadístico|diversas proteínas, entre ellas',re.I)

CURATED = [
    ('train','Ingeniería Genética_2 ed_2015.pdf',r'El DNA, al contrario que el RNA, es resistente a la hidrólisis en medio alcalino\.',
     'Compara DNA y RNA ante la hidrólisis alcalina.','comparación','replicación','DNA y RNA'),
    ('train','TUM - Biología molecular Fundamentos y aplicaciones en las ciencias de la salud - Salazar.pdf',r'La pentosa que compone el nucleótido es la D-ribosa\s+para el ARN y la D-desoxirribosa en el caso del ADN\.',
     'Compara la pentosa del ARN con la del ADN.','comparación','genética_molecular','pentosa ARN ADN'),
    ('train','TUM - Biología molecular Fundamentos y aplicaciones en las ciencias de la salud - Salazar.pdf',r'Las proteínas que provocan la fosforilación son las cinasas y las que eliminan los fosfatos son las fosfatasas\.',
     'Diferencia la acción de cinasas y fosfatasas.','diferenciación','genética_molecular','cinasas fosfatasas'),
    ('train','TUM - Biología molecular Fundamentos y aplicaciones en las ciencias de la salud - Salazar.pdf',r'Los\s+nucleótidos que contienen ribosa se denominan ribonucleótidos, mientras que los que contienen desoxirribosa,\s+desoxirribonucleótidos\.',
     'Diferencia ribonucleótidos y desoxirribonucleótidos.','diferenciación','genética_molecular','ribonucleótidos desoxirribonucleótidos'),
    ('train','Capítulo 14_ Vectores de clonación y expresión.pdf',r'Los vectores de expresión son aquellos cuyo objetivo es producir un tránscrito \(ARN\) o la proteína producto de ese tránscrito\.',
     'Si se desea producir un tránscrito, ¿qué vectores sirven?','aplicación_conceptual','clonación_vectores','vectores expresión'),
    ('train','Capítulo 13_ Enzimas de restricción.pdf',r'Las endonucleasas de restricción forman parte de la maquinaria con la que cuenta la bacteria para defenderse de infecciones virales\.',
     '¿Qué enzimas intervienen en la defensa bacteriana ante virus?','aplicación_conceptual','ingeniería_genética','endonucleasas restricción'),
    ('train','TUM - Biología molecular Fundamentos y aplicaciones en las ciencias de la salud - Salazar.pdf',r'La reacción de transcripción se\s+divide en tres etapas: iniciación, elongación y terminación\.',
     'Enumera las etapas de la transcripción.','etapas','transcripción_regulación','transcripción'),
    ('train','TUM - Biología molecular Fundamentos y aplicaciones en las ciencias de la salud - Salazar.pdf',r'El ciclo celular se divide en dos fases principales: la fase\s+M, o fase mitótica, y la interfase, o periodo preparatorio\.',
     'Identifica las fases principales del ciclo celular.','etapas','genética_molecular','ciclo celular'),
    ('validation','DNA recombinante y aplicaciones en medicina.pdf',r'Las técnicas Southern blot y Northern blot se utilizan para separar y caracterizar ADN y ARN\s+respectivamente\.',
     'Diferencia Southern blot y Northern blot.','diferenciación','ingeniería_genética','Southern Northern'),
    ('validation','DNA recombinante y aplicaciones en medicina.pdf',r'El corte con Bam HI \(izquierda\) produce extremos simple\s+cadena o “adhesivos” y el corte con AluI \(derecha\) produce extremos doble cadena o “romos”\.',
     'Compara los extremos producidos por Bam HI y AluI.','comparación','ingeniería_genética','Bam HI AluI'),
    ('validation','capitulo 88 biologia de las celulas madre.pdf',r'Las células multipotenciales \(p\. ej\., células madre hematopoyéticas\)\s+pueden formar múltiples linajes celulares pero no pueden dar origen a todos los linajes celulares del cuerpo\.',
     'Describe la capacidad y el límite de las células multipotenciales.','comprensión','células_madre','células multipotenciales'),
    ('validation','capitulo 88 biologia de las celulas madre.pdf',r'Las células oligopotenciales \(p\. ej\., células madre neurales\) pueden formar más de un linaje celular pero\s+están más restringidas que las células multipotenciales\.',
     '¿Cómo se compara la potencia de células oligopotenciales y multipotenciales?','comparación','células_madre','células oligopotenciales multipotenciales'),
    ('validation','capitulo 88 biologia de las celulas madre.pdf',r'Las células unipotenciales o las células monopotenciales \[p\. ej\., las células madre\s+progenitoras de espermatogonias \(SS\)\] pueden formar un único linaje celular diferenciado\.',
     'Describe qué linaje pueden formar las células unipotenciales.','comprensión','células_madre','células unipotenciales monopotenciales'),
]


def norm(s):
    return ' '.join(s.split())


def words(s):
    return re.findall(r'\w+',s.lower())


def key(s):
    return ' '.join(words(s))


def topic(s):
    return next((t for t,p in TOPICS.items() if re.search(p,s,re.I)), 'genética_molecular')


def terms(s):
    return {w for w in words(s) if w not in STOP and len(w)>2}


def anchors(s):
    # Detectar flexiones obvias sin asumir equivalencia científica entre sinónimos.
    return {w[:6] if len(w)>6 else w for w in terms(s)}


def broken_hyphen(s):
    allowed={'hélice-giro','giro-hélice','adeno-asociados','retro-transcripción','mesénquima-epitelio','ribonucleótido-difosfato'}
    return any(m[0].lower() not in allowed for m in re.finditer(r'\b[a-záéíóúñ]{2,}-[a-záéíóúñ]{2,}\b',s))


def grams(s):
    w=words(s)
    return set(zip(w,w[1:],w[2:]))


def clean_subject(s):
    return s[0].lower()+s[1:] if s.startswith(('El ','La ','Los ','Las ','Un ','Una ')) else s


def question(subject, verb, body, i):
    verb = re.sub(r'^(?:no|también|generalmente|normalmente|siempre|a menudo) ', '',verb.lower())
    if verb in ('se define como','se definen como') or verb in ('es','son') and re.match(r'(?:un|una|unos|unas) (?:proceso|mecanismo|colección|técnica|molécula|región|unidad|conjunto|agente|aminoácido|ciencia|campo|gráfico|enfoque|fragmento|constructo|portador|quimioterápico|tipo de)',body):
        category='definición'; options=[f'Define {subject} según el contexto.',f'Explica la definición de {subject}.',f'Explica el significado de {subject}.']
    elif verb.startswith(('permite','cataliza','protege','evita','codifica','sirve','desempeña','participa','interviene')):
        category='función'; options=[f'¿Cuál es la función de {subject}?',f'Describe el papel de {subject}.',f'Describe la función de {subject}.']
    elif verb.startswith(('depende','requiere','origina','genera','cambia','induce','aumenta','disminuye')):
        category='causa_consecuencia'; options=[f'¿Qué efecto o dependencia tiene {subject}?',f'Explica lo que causa o requiere {subject}.']
    elif verb.startswith(('se divide','se clasifica')):
        category='componentes'; options=[f'Describe la clasificación de {subject}.',f'¿Cómo se clasifica {subject} según el contexto?']
    elif verb.startswith(('contiene','incluye','comprende','está constitu','están constitu','está formado','están formado')):
        category='componentes'; options=[f'Describe la composición de {subject}.',f'¿Qué composición indica el contexto para {subject}?']
    elif verb.startswith(('se replica','se convierte','se sintetiza','se une','se divide','consiste','reconoce','sintetiza','transporta','actúa')):
        category='mecanismo'; options=[f'Explica cómo actúa {subject}.',f'Describe el proceso asociado a {subject}.']
    elif verb.startswith(('se asocia','corresponde','pertenece','regula')):
        category='relación'; options=[f'¿Qué relación describe el contexto para {subject}?',f'Relaciona {subject} con lo indicado en el contexto.']
    else:
        category=('explicación_breve','comprensión','interpretación','resumen')[i%4]
        options=[f'Explica lo indicado sobre {subject}.',f'¿Qué afirma el contexto sobre {subject}?',f'Interpreta la información sobre {subject}.',f'Resume el dato sobre {subject}.']
    q=options[i%len(options)]
    return category,re.sub(r'\bde el\b','del',re.sub(r'\ba el\b','al',q))


def extract(doc, tokenizer, rejected):
    result=[]
    pattern=re.compile(r'(?:^|(?<=[.!?])\s+)((?:El|La|Los|Las|Un|Una) [^.!?]{22,850}\.(?!\d))',re.M)
    for m in pattern.finditer(doc['text']):
        raw=m[1]; sentence=norm(raw)
        reason=None; relation=RELATION.fullmatch(sentence)
        if DEPENDENT.search(sentence) or NOISE.search(sentence) or REVIEW_NOISE.search(sentence) or broken_hyphen(sentence) or REVIEW_CLAIM.search(sentence) or sentence.count('(')!=sentence.count(')') or sentence.count('[')!=sentence.count(']'):
            reason='bad_extraction_or_missing_antecedent'
        elif not relation or BAD_SUBJECT.search(relation[1]) or AMBIGUOUS.fullmatch(relation[1]) or not SCIENCE.search(sentence):
            reason='insufficient_self_contained_concept'
        elif len(relation[1].split())>12 or re.search(r'\d{4}|\b(?:yo|usted)\b',sentence):
            reason='ambiguous_or_editorial_claim'
        if reason:
            rejected[reason]+=1; continue
        subject,verb,body=relation.groups()
        if REVIEW_SUBJECT.fullmatch(subject) or re.search(r'\b(?:otras?|a continuación|profesor|remanente|resultante|efectos más notorios)\b',subject,re.I):
            rejected['manual_review_missing_referent']+=1;continue
        answer=sentence; transformation='whitespace_only'
        # Proyección fiel de la primera cláusula: conserva sujeto/verbo y todos sus argumentos.
        # No se corta antes de un complemento ni dentro de paréntesis.
        if len(tokenizer.encode(answer))>42:
            cuts=[p.start() for p in re.finditer(r', (?:que |lo que |y |pero |por |debido |con |de modo |es decir |como )|;',answer)]
            shortened=next((answer[:c].rstrip()+'.' for c in cuts if answer[:c].count('(')==answer[:c].count(')') and RELATION.fullmatch(answer[:c].rstrip()+'.') and 15<=len(tokenizer.encode(answer[:c]+'.'))<=42),None)
            if shortened:
                answer=shortened; transformation='faithful_first_complete_clause'
        if not 12<=len(tokenizer.encode(answer))<=45:
            rejected['answer_budget']+=1; continue
        # Contexto entrega la afirmación autosuficiente, no la oración larga innecesaria.
        context=answer if transformation!='whitespace_only' else raw
        category,q=question(clean_subject(subject),verb,body,len(result))
        if verb.startswith('no '):
            category='comprensión';q=f'¿Qué afirma el contexto sobre {clean_subject(subject)}?'
        inferred_topic=topic(sentence)
        if '88 biologia de las celulas madre' in doc['source']:inferred_topic='células_madre'
        item=dict(source_doc=doc['source'],source_family=doc['family'],source_split=doc['split'],source_sha256=doc['sha256'],
                  evidence_text=raw,evidence_start=m.start(1),evidence_end=m.end(1),context=context,answer=answer,
                  question=q,subject=clean_subject(subject),verb=verb,topic=inferred_topic,instruction_category=category,
                  transformation=transformation,claim_key=key(answer))
        result.append(item)
    return result


def token_fields(r,tok):
    r.update(context_token_count=len(tok.encode(r['context'])),question_token_count=len(tok.encode(r['question'])),
             answer_token_count=len(tok.encode(r['answer'])),total_token_count=2+len(tok.encode(prompt(r)+r['answer'])))


def positive(item,split,index,tok,task='grounded_answerable'):
    r=dict(item,id=f'{split}-v2-{index:05d}',split=split,task_type=task,supported=True,difficulty='sencilla',pair_id=None,
           evidence_sources=[dict(source_doc=item['source_doc'],source_split=split,source_sha256=item['source_sha256'],
               start=item['evidence_start'],end=item['evidence_end'],text=item['evidence_text'])])
    if task=='direct_instruction':
        r['context']='';r['question']=r['question'].replace(' según el contexto','').replace('el contexto','la evidencia documental')
    token_fields(r,tok)
    return r


def compatible_negative(pos, other):
    if pos['topic']!=other['topic'] or pos['claim_key']==other['claim_key']:
        return False
    domains=[r'\b(?:ADN|DNA)\b',r'\b(?:ARN\w*|RNA\w*)\b',r'\bgen(?:es|oma|ético|ética|ómicas|ómico)?\b',
             r'célul|celular',r'proteín|proteic',r'enzima|polimerasa|cinasa',r'virus|viral',
             r'cromosom|alelo|herencia',r'plásmid|vector|clonaci',r'histon|metil|epigen',r'PCR|secuenci|restricci']
    if not any(re.search(p,pos['context'],re.I) and re.search(p,other['context'],re.I) for p in domains):
        return False
    # El contexto no nombra la entidad concreta solicitada y habla de otra relación.
    anchor=terms(pos['subject'])
    if not anchor or len(anchor)<1:
        return False
    context_terms=terms(other['context'])
    if anchor & context_terms or anchors(pos['subject']) & anchors(other['context']):
        return False
    # Tampoco puede contener la mayoría de los argumentos que resolverían la pregunta.
    answer_terms=terms(pos['answer'])
    if len(answer_terms & context_terms)/max(1,len(answer_terms))>.15:
        return False
    return True


def negative(pos,other,split,index,tok,pair_id):
    r=dict(other,id=f'{split}-v2-{index:05d}',split=split,task_type='abstention',supported=False,difficulty='contrastiva',
           pair_id=pair_id,question=pos['question'],answer=ABSTENTIONS[index%2],
           instruction_category=pos['instruction_category'],requested_topic=pos['topic'],requested_subject=pos['subject'],
           positive_source_doc=pos['source_doc'],positive_reference=pos['answer'],
           negative_reason='El fragmento trata el mismo tema pero solo afirma una relación sobre otra entidad; no contiene la entidad concreta ni sus argumentos solicitados.',
           negative_anchor_terms=sorted(terms(pos['subject'])),
           evidence_sources=[dict(source_doc=other['source_doc'],source_split=split,source_sha256=other['source_sha256'],
               start=other['evidence_start'],end=other['evidence_end'],text=other['evidence_text'])])
    token_fields(r,tok)
    return r


def audit(r,by_doc,tok):
    doc=by_doc[r['source_doc']]
    require(r['source_split']==r['split']==doc['split'], 'Split incorrecto.')
    require(doc['text'][r['evidence_start']:r['evidence_end']]==r['evidence_text'],'Span no literal.')
    require((r.get('curated') or not DEPENDENT.search(norm(r['evidence_text']))) and not NOISE.search(norm(r['evidence_text'])) and not REVIEW_NOISE.search(norm(r['evidence_text'])) and not broken_hyphen(norm(r['evidence_text'])),'Extracción dependiente/rota.')
    if r['supported']:
        original=norm(r['evidence_text'])
        require(r['answer']==original or original.startswith(r['answer'][:-1]) and r['transformation']=='faithful_first_complete_clause','Hechos no respaldados.')
        require(r['answer']==r['answer'].strip() and r['answer'][0].isupper() and r['answer'].endswith('.'),'Inicio/cierre incompleto.')
        if not r.get('curated'):
            require(RELATION.fullmatch(r['answer']) is not None,'Afirmación no autónoma.')
    else:
        require(r['answer'] in ABSTENTIONS,'Abstención no canónica.')
        require(not (set(r['negative_anchor_terms']) & terms(r['context'])),'Negativo menciona entidad solicitada.')
        require(not (anchors(r['requested_subject']) & anchors(r['context'])),'Negativo menciona flexión de entidad solicitada.')
        require(r['positive_reference']!=norm(r['context']),'Negativo contiene respuesta.')
    encode(r,tok)
    return dict(id=r['id'],pair_id=r['pair_id'],source_doc=r['source_doc'],source_split=r['source_split'],
                supported=r['supported'],evidence_span_exact=True,answer_fidelity=True if r['supported'] else None,
                negative_sufficiency_checked=not r['supported'],negative_reason=r.get('negative_reason'),
                method='extractive_clause_support' if r['supported'] else 'same_topic_distinct_entity_and_missing_arguments',
                evidence_start=r['evidence_start'],evidence_end=r['evidence_end'],total_token_count=r['total_token_count'],passed=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--draft',action='store_true');mode.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    require(not (DATA/'dataset_lock_5A1.json').exists(),'Dataset v2 ya congelado; no reconstruir.')
    verify_preservation(); docs=documents();tok=TokenizadorEducativo().cargar()
    rejected=Counter();dedup=Counter();output={};pair_rows=[];audits=[];seen_questions=set();seen_claims=set();near=[]
    pool_counts={}
    for split in ('train','validation'):
        by_doc={d['source']:d for d in docs[split]}
        pools={d['source']:extract(d,tok,rejected) for d in docs[split]}
        pool_counts[split]={s:len(v) for s,v in pools.items()}
        all_items=[i for p in pools.values() for i in p]
        for source,pool in pools.items():
            random.Random(SEED+int(hashlib.sha256(source.encode()).hexdigest()[:8],16)).shuffle(pool)
        rows=[];doc_counts=Counter();positive_items=[]
        for csplit,source,expression,q,category,ctopic,subject in CURATED:
            if csplit!=split:continue
            doc=by_doc[source];m=re.search(expression,doc['text']);require(m is not None,'Evidencia curada no encontrada: '+q)
            item=dict(source_doc=source,source_family=doc['family'],source_split=split,source_sha256=doc['sha256'],
                      evidence_text=m[0],evidence_start=m.start(),evidence_end=m.end(),context=m[0],answer=norm(m[0]),
                      question=q,subject=subject,verb='curated',topic=ctopic,instruction_category=category,
                      transformation='whitespace_only',claim_key=key(m[0]),curated=True)
            r=positive(item,split,len(rows),tok);audit(r,by_doc,tok);rows.append(r);positive_items.append(r)
            doc_counts[source]+=1;seen_questions.add(key(q));seen_claims.add(item['claim_key']);near.append(grams(q));all_items.append(item)
        limit=640 if split=='train' else 100
        cap=75 if split=='train' else 42
        while any(pools.values()) and len(positive_items)<limit:
            for source in sorted(pools):
                if doc_counts[source]>=cap or len(positive_items)>=limit:
                    rejected['quota_unselected_candidates']+=len(pools[source]);pools[source].clear();continue
                if not pools[source]:continue
                item=pools[source].pop();qkey=key(item['question']);ckey=item['claim_key'];g=grams(item['question'])
                if qkey in seen_questions or ckey in seen_claims:
                    dedup['exact_duplicates_removed']+=1;continue
                if any(len(g&p)/max(1,len(g|p))>=.8 for p in near):
                    dedup['near_duplicates_removed']+=1;continue
                r=positive(item,split,len(rows),tok)
                try: audit(r,by_doc,tok)
                except RuntimeError as e: rejected[str(e)]+=1;continue
                rows.append(r);positive_items.append(r);doc_counts[source]+=1
                seen_questions.add(qkey);seen_claims.add(ckey);near.append(g)
        # 55:5:25 por registros aproximados, y mezcla por pasos exacta en schedule().
        pair_target=max(150 if split=='train' else max(35,120-len(rows)),math.ceil(len(rows)*25/60))
        pair_candidates=sorted(positive_items,key=lambda p:(doc_counts[p['source_doc']],p['id']))
        pairs_created=0
        negative_doc_counts=Counter()
        negative_context_reuse=Counter()
        for pos in pair_candidates:
            if pairs_created>=pair_target:break
            choices=[i for i in all_items if compatible_negative(pos,i) and negative_context_reuse[i['claim_key']]<3]
            # Prioriza el mismo documento, salvo concentración de cuotas o inexistencia.
            choices.sort(key=lambda i:(doc_counts[i['source_doc']]+negative_doc_counts[i['source_doc']], i['source_doc']!=pos['source_doc'],i['claim_key']))
            for other in choices:
                pair_id=f'{split}-pair-{pairs_created:04d}'
                neg=negative(pos,other,split,len(rows),tok,pair_id)
                try:audit(neg,by_doc,tok)
                except RuntimeError:continue
                pos['pair_id']=pair_id;rows.append(neg);negative_doc_counts[other['source_doc']]+=1;negative_context_reuse[other['claim_key']]+=1
                pair_rows.append(dict(pair_id=pair_id,split=split,positive_id=pos['id'],negative_id=neg['id'],question=pos['question'],
                                      positive_context=pos['context'],negative_context=neg['context'],positive_answer=pos['answer'],negative_answer=neg['answer'],
                                      positive_source_doc=pos['source_doc'],negative_source_doc=neg['source_doc'],topic=pos['topic'],audit_passed=True))
                pairs_created+=1;break
        # Direct: reserva 1/12 de los positivos que no forman par y sean definiciones claras.
        direct_target=round(len(positive_items)/12)
        corpus_words=' '.join(key(d['text']) for d in docs['train'])
        direct_candidates=[r for r in positive_items if r['pair_id'] is None and r['instruction_category']=='definición' and 'energéticamente' not in r['answer']
                           and corpus_words.count(re.sub(r'^(?:el|la|los|las|un|una) ','',key(r['subject'])))>=4]
        for r in direct_candidates[:direct_target]:
            r['task_type']='direct_instruction';r['context']='';r['question']=r['question'].replace(' según el contexto','')
            token_fields(r,tok)
        # Una categoría con un único registro no se fuerza: se agrupa como comprensión.
        if split=='train':
            category_counts=Counter(r['instruction_category'] for r in rows)
            for r in rows:
                if category_counts[r['instruction_category']]==1:r['instruction_category']='comprensión'
        output[split]=rows
    # Segunda auditoría completa antes de publicar propuesta/congelar.
    for split,rows in output.items():
        by_doc={d['source']:d for d in docs[split]}
        audits.extend(audit(r,by_doc,tok) for r in rows)
    require(not ({key(r['question']) for r in output['train']}&{key(r['question']) for r in output['validation']}),'Pregunta compartida Train/Validation.')
    for split,rows in output.items():
        (DATA/f'{split}_instrucciones_v2.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    for name,rows in [('auditoria_evidencia',audits),('pares_contrastivos',pair_rows)]:
        (DATA/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    validation=output['validation'];vpairs=[p for p in pair_rows if p['split']=='validation']
    require(len(vpairs)>=12,'Validation no permite 12 pares para suite fija.')
    # Suite: 24 positivos y 24 negativos, de los cuales 12 pares exactos.
    by_id={r['id']:r for r in validation};chosen=[by_id[p[k]] for p in vpairs[:12] for k in ('positive_id','negative_id')]
    used={r['id'] for r in chosen}
    for supported in (True,False):
        extras=[r for r in validation if r['supported']==supported and r['id'] not in used and r['task_type']!='direct_instruction'][:12]
        require(len(extras)==12,'Suite necesita 24 positivos/24 negativos únicos.')
        chosen.extend(extras);used.update(r['id'] for r in extras)
    suite=dict(version='5A.1',seed=42,selection='12 first contrastive pairs plus first unused positives/negatives; fixed before training',
               generation_config=dict(greedy=True,max_new_tokens=48,seed=42),records=chosen,pair_ids=[p['pair_id'] for p in vpairs[:12]])
    write_json(DATA/'suite_validation_5A1.json',suite)
    train=output['train'];max_doc=max(Counter(r['source_doc'] for r in train).values())/len(train)
    two_doc=sum(sorted(Counter(r['source_doc'] for r in train).values(),reverse=True)[:2])/len(train)
    warnings=[]
    if len(train)<800:warnings.append('Train debajo de 800: constructor de cláusulas explícitas, presupuesto 128, deduplicación y cuotas; no prueba que las fuentes no admitan más curación humana.')
    if len(validation)<120:warnings.append('Validation debajo de 120; no se infló con paráfrasis.')
    if max_doc>.18:warnings.append('WARNING: un documento supera 18%; revisar cuotas y disponibilidad de evidencia.')
    if two_doc>.35:warnings.append('WARNING: dos documentos superan 35%.')
    counts={s:dict(Counter(r['task_type'] for r in rows)) for s,rows in output.items()}
    report=dict(version='5A.1',constructor_version=VERSION,seed=42,status='frozen' if args.freeze else 'draft',
        train_examples=len(train),validation_examples=len(validation),counts_by_task_type=counts,
        examples_by_document={s:{d['source']:dict(count=sum(r['source_doc']==d['source'] for r in rows),percent=100*sum(r['source_doc']==d['source'] for r in rows)/len(rows)) for d in docs[s]} for s,rows in output.items()},
        examples_by_topic={s:dict(Counter(r['topic'] for r in rows)) for s,rows in output.items()},
        examples_by_category={s:dict(Counter(r['instruction_category'] for r in rows)) for s,rows in output.items()},
        answerable_count={s:sum(r['supported'] for r in rows) for s,rows in output.items()},
        abstention_count={s:sum(not r['supported'] for r in rows) for s,rows in output.items()},
        contrastive_pair_count={s:sum(p['split']==s for p in pair_rows) for s in output},
        token_statistics={s:{'mean_context_tokens':mean(r['context_token_count'] for r in rows),'mean_question_tokens':mean(r['question_token_count'] for r in rows),
                            'mean_answer_tokens':mean(r['answer_token_count'] for r in rows),'max_total_tokens':max(r['total_token_count'] for r in rows)} for s,rows in output.items()},
        deduplication=dict(dedup),rejected_examples=dict(rejected),candidate_counts_by_document=pool_counts,
        exact_duplicates_removed=dedup['exact_duplicates_removed'],near_duplicates_removed=dedup['near_duplicates_removed'],
        rejected_for_insufficient_evidence=rejected['insufficient_self_contained_concept'],
        rejected_for_bad_extraction=rejected['bad_extraction_or_missing_antecedent'],
        rejected_for_context_budget=rejected['answer_budget']+rejected['Presupuesto/specials/IDs inválidos; nunca truncar.'],
        test_documents_used=0,test_exclusion_verified=True,all_answerable_with_evidence=True,all_negatives_audited=True,
        balance=dict(max_document_percent=100*max_doc,top_two_document_percent=100*two_doc),
        warnings=warnings,hashes=hashes(False),base_checkpoint_sha256=sha256(BASE/'modelos/checkpoint_best.pt'),
        previous_experiment_preserved=True,negative_audit_limitations='Insuficiencia contrastiva se comprueba para afirmaciones explícitas sobre entidades distintas del mismo tema; no equivale a prueba universal de ausencia de inferencias científicas.',
        intentional_duplicate_questions='Solo miembros positive/negative del mismo pair_id. Se prohíben preguntas iguales entre Train y Validation.')
    write_json(DATA/'reporte_dataset_sft_5A1.json',report)
    # Muestra humana estratificada: 40 Train, 30 Validation, semilla fija.
    sections=['# Auditoría de dataset 5A.1','\nPropuesta contrastiva de fuentes Train/Validation originales. No se usó Test.\n',
              'Las respuestas son cláusulas completas respaldadas; no se añaden hechos externos. El contexto negativo comparte tema pero carece de la entidad/relación solicitada.\n']
    for split,n in [('train',40),('validation',30)]:
        rows=output[split];rng=random.Random(42);sample=[]
        for task in TASKS:
            pool=[r for r in rows if r['task_type']==task];rng.shuffle(pool);sample.extend(pool[:max(2,n//3)])
        used={r['id'] for r in sample};remaining=[r for r in rows if r['id'] not in used];rng.shuffle(remaining);sample=(sample+remaining)[:n]
        sections.append(f'\n## {split}: {len(sample)} registros\n')
        for r in sample:
            sections.append(f"\n### {r['id']} — {r['task_type']}\n\nFuente: {r['source_doc']} · Tema: {r['topic']} · Categoría: {r['instruction_category']} · Par: {r['pair_id']}\n\n**Contexto:** {r['context'] or '(No visible: instrucción directa)'}\n\n**Evidencia:** {norm(r['evidence_text'])}\n\n**Pregunta:** {r['question']}\n\n**Respuesta:** {r['answer']}\n")
    (RESULTS/'auditoria_dataset_sft_5A1.md').write_text('\n'.join(sections),encoding='utf-8')
    if args.freeze:
        write_json(DATA/'dataset_lock_5A1.json',dict(version='5A.1',hashes=hashes(False),constructor_version=VERSION,seed=42))
    verify_preservation();log_accesses('preparacion')
    print(json.dumps({k:report[k] for k in ('train_examples','validation_examples','contrastive_pair_count','counts_by_task_type','examples_by_category','balance','warnings','candidate_counts_by_document')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
