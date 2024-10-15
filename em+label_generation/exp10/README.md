# exp10

-----
## Experiment description
Try classifier free guidance by conditioning on dataset id

## License

`exp10` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Data Preparation

`hatch run data-prep filter-yaml zarrdata.yaml exp10data.yaml --scale 8 8 8 --min-size 1 96 96 --min-frac-annotated 1 --labels ecs --labels pm --labels mito_mem --labels mito_lum --labels mito_ribo --labels golgi_mem --labels golgi_lum --labels ves_mem --labels ves_lum --labels endo_mem --labels endo_lum --labels lyso_mem --labels lyso_lum --labels ld_mem --labels ld_lum --labels er_mem --labels er_lum --labels eres_mem --labels eres_lum --labels ne_mem --labels ne_lum --labels np_out --labels np_in --labels hchrom --labels nhchrom --labels echrom --labels nechrom --labels nucpl --labels nucleo --labels mt_out --labels mt_in`