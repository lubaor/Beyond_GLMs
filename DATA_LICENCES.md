# Data licences and attribution

One entry per source. All URLs were resolved and confirmed live on
**2026-08-28**, and `fetch_data.py` was run end to end against them on the same
date: every dataset reproduced content-identical to the files this benchmark
was built on.

**No dataset is redistributed in this repository.** `fetch_data.py` downloads
each one from the provider at run time. The Apache-2.0 licence in `LICENSE`
covers the code here and nothing else. Each dataset remains under its own
terms, below.

Every source listed here has a licence that is stated and open. A sixth
dataset, used for the new-business conversion step, is deliberately absent:
see the note at the end.

---

## 1. freMTPL2freq — French motor, claim frequency

| | |
|---|---|
| Provider | OpenML |
| Product | `freMTPL2freq`, data id 41214, version 1, ARFF |
| Licence | CC0 1.0 (public domain dedication) |
| Attribution | Originally distributed with the R package CASdatasets; mirrored on OpenML under CC0 |
| Landing page | https://www.openml.org/d/41214 |
| Download | via `sklearn.datasets.fetch_openml(data_id=41214)` |
| Local file | `01_freMTPL2_freq.csv`, 678,013 rows |
| Retrieved | 2026-08-28 |

## 2. freMTPL2sev — French motor, claim severities

| | |
|---|---|
| Provider | Christophe Dutang and Arthur Charpentier |
| Product | CASdatasets 1.2-1, dataset `freMTPL2sev` |
| Licence | [GPL (>= 2)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html) |
| Attribution | Dutang C., Charpentier A., CASdatasets R package version 1.2-1 |
| Landing page | https://dutangc.github.io/CASdatasets/ |
| Download | https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/freMTPL2sev.rda |
| Local file | `02_freMTPL2_sev.csv`, 26,444 rows |
| Retrieved | 2026-08-28 |

## 3. swmotorcycle — Swedish Wasa motorcycle

| | |
|---|---|
| Provider | Christophe Dutang and Arthur Charpentier |
| Product | CASdatasets 1.2-1, dataset `swmotorcycle` |
| Licence | [GPL (>= 2)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html) |
| Attribution | Dutang C., Charpentier A., CASdatasets 1.2-1. Underlying data from the former Swedish insurer Wasa, 1994 to 1998, via Ohlsson and Johansson |
| Landing page | https://dutangc.github.io/CASdatasets/reference/swmotorcycle.html |
| Download | https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/swmotorcycle.rda |
| Local file | `03_wasa_motorcycle.csv`, 64,548 rows (62,474 modelled) |
| Retrieved | 2026-08-28 |

## 4. ausprivauto0405 — Australian private motor

| | |
|---|---|
| Provider | Christophe Dutang and Arthur Charpentier |
| Product | CASdatasets 1.2-1, dataset `ausprivauto0405` |
| Licence | [GPL (>= 2)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html) |
| Attribution | Dutang C., Charpentier A., CASdatasets 1.2-1. Underlying data from De Jong P. and Heller G.Z., *Generalized Linear Models for Insurance Data* |
| Landing page | https://dutangc.github.io/CASdatasets/ |
| Download | https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/ausprivauto0405.rda |
| Local file | `04_ausprivauto_dejong.csv`, 67,856 rows |
| Retrieved | 2026-08-28 |

## 5. eudirectlapse — European renewal lapse

| | |
|---|---|
| Provider | Christophe Dutang and Arthur Charpentier |
| Product | CASdatasets 1.2-1, dataset `eudirectlapse` |
| Licence | [GPL (>= 2)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html) |
| Attribution | Dutang C., Charpentier A., CASdatasets R package version 1.2-1 |
| Landing page | https://dutangc.github.io/CASdatasets/ |
| Download | https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/eudirectlapse.rda |
| Local file | `05_eudirectlapse_demand.csv`, 23,060 rows |
| Retrieved | 2026-08-28 |

---

## A sixth dataset, excluded

The talk this repository accompanies also covers a new-business conversion
example. That step is **not** included here, because it depends on a dataset
whose licence terms are unresolved: a tutorial file hosted by ActiveViam for
the Atoti notebook gallery, with no licence stated at the data URL. The Atoti
notebook repository is Apache-2.0, but that covers the notebooks rather than
the hosted CSV.

Rather than publish code that points at a source we cannot license cleanly,
that step is kept out of this repository. Nothing else in the benchmark
depends on it, and the conclusions in steps 01 to 09 stand without it.

---

## Licence compatibility

Four of the five datasets are GPL (>= 2). The code in this repository is
Apache-2.0, which is not compatible with GPLv2 for the purpose of combining
into a single distributed work.

That does not arise here. The code neither incorporates nor redistributes any
CASdatasets material. It reads CSV files that the user downloads themselves at
run time, which is use rather than distribution. Anyone who goes on to
redistribute the downloaded data, rather than the fetch script, takes on the
upstream obligations directly.
