# Perfil de material (transcripts)

GENERADO por `tools/perfil_transcript.py`. Nada escrito a mano.


Se lee antes que cualquier conclusion de #72: un histograma de huecos construido sobre timestamps que no son de fiar describe el modelo, no el audio.

## STEP 1 — Auditoria de las transcripciones


| transcript | palabras | span s | span | veredicto |
|---|---|---|---|---|
| kick_test | 22697 | 11247 | 187 min | parcialmente confiable |
| youtube_test | 18582 | 4939 | 82 min | confiable |

### `kick_test`
Ruta: `runs/kick_test/package/delivery-work/transcript.json`. Idioma: `es`.

**(a) Porcion sin voz transcripta**
- Huecos entre palabras: 22696. Huecos > 1.0 s: **814** = **2446 s** = **21.8 %** del span.
- Huecos > 2,0 s: 317 = 1754 s.
- Huecos > 3,0 s: 185 = 1437 s.
- Huecos negativos (palabras solapadas): 0.

**(b) Alucinaciones de Whisper**
| frase canonica | ocurrencias |
|---|---|
| gracias por ver | 35 |
| gracias por ver el video | 35 |

Primer bloque: `gracias por ver` en t=28.5 s, confianza de la primera palabra 0.080.

- Palabras con confianza < 0.3: **1405** (6.19 %), **607 s** de duracion atribuida.

| texto | t (s) | confianza |
|---|---|---|
| Gracias | 28.5 | 0.080 |
| Gracias | 44.8 | 0.066 |
| Gracias | 71.7 | 0.055 |
| Gracias | 90.0 | 0.026 |
| Gracias | 121.2 | 0.032 |

**(c) Palabras con duracion imposible**
- Mediana de duracion de palabra: **0.2800 s**, desviacion tipica **0.5087 s**.
- A mas de 3 desviaciones: cortas **0**, largas **75**.
- Duracion <= 0: **106** (0.5 %). Duracion >= 1,0 s: **658**.

**(d) Monotonia**
- Retrocesos de `start_s`: **0**.

**(e) Puntuacion**
- Palabras cerradas con puntuacion: **0.55 %** de las palabras.

| signo | palabras |
|---|---|
| ! | 1 |
| , | 23 |
| . | 3 |
| ? | 97 |

- Cierres de frase (`.` `?` `!`): **101** = 0.5 por minuto.
- Distancia entre cierres consecutivos: mediana **20.69 s**, media 101.87 s, min 0.42 s, max 1112.68 s.

### `youtube_test`
Ruta: `runs/youtube_test/package/delivery-work/transcript.json`. Idioma: `es`.

**(a) Porcion sin voz transcripta**
- Huecos entre palabras: 18581. Huecos > 1.0 s: **128** = **260 s** = **5.3 %** del span.
- Huecos > 2,0 s: 35 = 136 s.
- Huecos > 3,0 s: 21 = 102 s.
- Huecos negativos (palabras solapadas): 0.

**(b) Alucinaciones de Whisper**
- Ninguna frase canonica.

- Palabras con confianza < 0.3: **413** (2.22 %), **82 s** de duracion atribuida.

| texto | t (s) | confianza |
|---|---|---|
| el | 31.6 | 0.157 |
| uh, | 48.5 | 0.187 |
| Real | 73.7 | 0.181 |
| nada, | 78.9 | 0.263 |
| O | 94.2 | 0.018 |

**(c) Palabras con duracion imposible**
- Mediana de duracion de palabra: **0.1600 s**, desviacion tipica **0.1442 s**.
- A mas de 3 desviaciones: cortas **0**, largas **360**.
- Duracion <= 0: **284** (1.5 %). Duracion >= 1,0 s: **46**.

**(d) Monotonia**
- Retrocesos de `start_s`: **0**.

**(e) Puntuacion**
- Palabras cerradas con puntuacion: **23.28 %** de las palabras.

| signo | palabras |
|---|---|
| ! | 1 |
| , | 2862 |
| . | 1253 |
| ? | 210 |

- Cierres de frase (`.` `?` `!`): **1464** = 17.8 por minuto.
- Distancia entre cierres consecutivos: mediana **1.74 s**, media 3.08 s, min 0.00 s, max 89.32 s.

## STEP 2 — Histograma de huecos sobre estos transcripts


### `kick_test` — 22696 huecos
| cubo (s) | n | segundos |
|---|---|---|
| <0.2 | 21182 | 5.4 |
| 0.2-0.5 | 59 | 23.0 |
| 0.5-1.0 | 629 | 464.5 |
| 1.0-1.5 | 337 | 407.7 |
| 1.5-2.0 | 169 | 290.7 |
| 2.0-3.0 | 135 | 323.4 |
| >3.0 | 185 | 1436.6 |

Veredicto de integridad: **parcialmente confiable**.

### `youtube_test` — 18581 huecos
| cubo (s) | n | segundos |
|---|---|---|
| <0.2 | 16886 | 259.8 |
| 0.2-0.5 | 1135 | 371.4 |
| 0.5-1.0 | 425 | 281.9 |
| 1.0-1.5 | 73 | 85.4 |
| 1.5-2.0 | 27 | 45.8 |
| 2.0-3.0 | 13 | 30.8 |
| >3.0 | 22 | 104.7 |

Veredicto de integridad: **confiable**.

## STEP 3 — La tabla que decide
Un tramo cuenta como utilizable a partir de **8 s**. Gana el mecanismo que captura MAS TIEMPO en tramos utilizables, no el que hace mas tramos.

### `kick_test`


**Por mecanismo, aislado**
| mecanismo | tramos | min | mediana | max | capturada s |
|---|---|---|---|---|---|
| MAX_GAP_S=0.6 | 1357 | 0.1 | 4.0 | 65.6 | 8378.3 |
| MAX_GAP_S=1.0 | 815 | 0.1 | 6.6 | 84.2 | 8800.6 |
| MAX_GAP_S=1.5 | 482 | 0.1 | 11.2 | 231.4 | 9208.4 |
| MAX_GAP_S=2.0 | 318 | 0.1 | 13.1 | 376.4 | 9493.0 |
| MAX_GAP_S=3.0 | 186 | 0.1 | 17.3 | 980.4 | 9810.4 |

| mecanismo | tramos | min | mediana | max | capturada s |
|---|---|---|---|---|---|
| PUNCT_BREAK_S=10 | 76 | 1.0 | 44.2 | 1112.8 | 11247.0 |
| PUNCT_BREAK_S=15 | 69 | 1.0 | 53.9 | 1112.8 | 11247.0 |
| PUNCT_BREAK_S=20 | 63 | 1.0 | 85.5 | 1112.8 | 11247.0 |
| PUNCT_BREAK_S=30 | 55 | 1.0 | 104.3 | 1112.8 | 11247.0 |
| PUNCT_BREAK_S=45 | 49 | 1.0 | 112.4 | 1112.8 | 11247.0 |
| PUNCT_BREAK_S=None | 102 | 0.7 | 21.5 | 1112.8 | 11247.0 |

**Cruzada**
| MAX_GAP_S | PUNCT_BREAK_S | tramos | utiles | segundos utiles |
|---|---|---|---|---|
| 0.6 | 10 | 1365 | 347 | 5098 |
| 0.6 | 15 | 1361 | 345 | 5100 |
| 0.6 | 20 | 1360 | 345 | 5106 |
| 0.6 | 30 | 1358 | 344 | 5107 |
| 0.6 | 45 | 1358 | 343 | 5106 |
| 0.6 | None | 1357 | 343 | 5107 |
| 1.0 | 10 | 843 | 381 | 7205 |
| 1.0 | 15 | 833 | 377 | 7231 |
| 1.0 | 20 | 830 | 373 | 7226 |
| 1.0 | 30 | 822 | 371 | 7251 |
| 1.0 | 45 | 820 | 366 | 7241 |
| 1.0 | None | 815 | 364 | 7251 |
| 1.5 | 10 | 528 | 323 | 8469 |
| 1.5 | 15 | 518 | 315 | 8479 |
| 1.5 | 20 | 508 | 308 | 8499 |
| 1.5 | 30 | 499 | 304 | 8517 |
| 1.5 | 45 | 496 | 301 | 8523 |
| 1.5 | None | 482 | 288 | 8524 |
| 2.0 | 10 | 370 | 259 | 9090 |
| 2.0 | 15 | 365 | 254 | 9092 |
| 2.0 | 20 | 358 | 248 | 9099 |
| 2.0 | 30 | 345 | 236 | 9100 |
| 2.0 | 45 | 340 | 231 | 9100 |
| 2.0 | None | 318 | 211 | 9109 |
| 3.0 | 10 | 250 | 192 | 9615 |
| 3.0 | 15 | 242 | 184 | 9612 |
| 3.0 | 20 | 237 | 180 | 9618 |
| 3.0 | 30 | 226 | 169 | 9618 |
| 3.0 | 45 | 220 | 163 | 9618 |
| 3.0 | None | 186 | 129 | 9618 |

### `youtube_test`


**Por mecanismo, aislado**
| mecanismo | tramos | min | mediana | max | capturada s |
|---|---|---|---|---|---|
| MAX_GAP_S=0.6 | 387 | 0.2 | 6.9 | 89.9 | 4483.8 |
| MAX_GAP_S=1.0 | 129 | 0.3 | 22.8 | 272.1 | 4678.8 |
| MAX_GAP_S=1.5 | 62 | 0.5 | 48.6 | 380.8 | 4758.7 |
| MAX_GAP_S=2.0 | 36 | 0.5 | 100.6 | 391.4 | 4803.0 |
| MAX_GAP_S=3.0 | 22 | 21.1 | 116.4 | 1116.4 | 4836.9 |

| mecanismo | tramos | min | mediana | max | capturada s |
|---|---|---|---|---|---|
| PUNCT_BREAK_S=10 | 354 | 10.0 | 12.1 | 97.1 | 4938.5 |
| PUNCT_BREAK_S=15 | 256 | 15.0 | 17.1 | 97.1 | 4938.5 |
| PUNCT_BREAK_S=20 | 199 | 20.0 | 22.9 | 97.1 | 4938.5 |
| PUNCT_BREAK_S=30 | 142 | 30.0 | 32.0 | 117.0 | 4938.5 |
| PUNCT_BREAK_S=45 | 99 | 45.1 | 47.9 | 89.5 | 4938.5 |
| PUNCT_BREAK_S=None | 1465 | 0.1 | 2.0 | 89.5 | 4938.5 |

**Cruzada**
| MAX_GAP_S | PUNCT_BREAK_S | tramos | utiles | segundos utiles |
|---|---|---|---|---|
| 0.6 | 10 | 564 | 287 | 3574 |
| 0.6 | 15 | 494 | 244 | 3652 |
| 0.6 | 20 | 455 | 220 | 3734 |
| 0.6 | 30 | 412 | 197 | 3803 |
| 0.6 | 45 | 393 | 186 | 3825 |
| 0.6 | None | 387 | 181 | 3831 |
| 1.0 | 10 | 415 | 328 | 4360 |
| 1.0 | 15 | 318 | 256 | 4449 |
| 1.0 | 20 | 271 | 209 | 4453 |
| 1.0 | 30 | 214 | 169 | 4526 |
| 1.0 | 45 | 177 | 141 | 4561 |
| 1.0 | None | 129 | 97 | 4577 |
| 1.5 | 10 | 381 | 342 | 4621 |
| 1.5 | 15 | 286 | 253 | 4648 |
| 1.5 | 20 | 232 | 203 | 4658 |
| 1.5 | 30 | 173 | 153 | 4688 |
| 1.5 | 45 | 137 | 116 | 4687 |
| 1.5 | None | 62 | 53 | 4742 |
| 2.0 | 10 | 363 | 338 | 4708 |
| 2.0 | 15 | 269 | 252 | 4732 |
| 2.0 | 20 | 218 | 200 | 4734 |
| 2.0 | 30 | 162 | 144 | 4726 |
| 2.0 | 45 | 118 | 105 | 4740 |
| 2.0 | None | 36 | 32 | 4789 |
| 3.0 | 10 | 358 | 344 | 4779 |
| 3.0 | 15 | 265 | 254 | 4790 |
| 3.0 | 20 | 212 | 201 | 4794 |
| 3.0 | 30 | 153 | 142 | 4784 |
| 3.0 | 45 | 111 | 105 | 4805 |
| 3.0 | None | 22 | 22 | 4837 |

## STEP 4 — Lo que NO se midio
**Delta de energia: BLOQUEADO.** Requiere audio. `astats` lee muestras y este material es solo word timestamps de Whisper. No hay cifra de RMS, de pico ni de variacion aqui, y no se estima.

Tambien sin medir, por falta de audio: huecos que contienen voz superpuesta, numero de hablantes, y si un silencio va seguido de grito o risa. Los histogramas de arriba tratan TODO hueco grande como silencio, y en material de stream eso es una aproximacion que puede estar equivocada.

