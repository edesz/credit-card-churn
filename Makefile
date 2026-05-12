#################################################################################
# GLOBALS                                                                       #
#################################################################################

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Run Jupyterlab
jlab:
	@echo "+ $@"
	@pixi run jlab
.PHONY: jlab

## Run notebooks
nbs:
	@echo "+ $@"
	@pixi run nbs -- 10,11,12
.PHONY: nbs

## Run data validation with Pandera
dval:
	@echo "+ $@"
	@pixi run nbs -- 12
.PHONY: dval

## Run ML Validation experiments with Metaflow
mlval:
	@echo "+ $@"
	@pixi run nbs -- 04
.PHONY: mlval

## Run unit tests with Pytest
tests:
	@echo "+ $@"
	@pixi run test
.PHONY: tests

## Upgrade conda package versions with pxi
pixi-upgrade:
	@echo "+ $@"
	@pixi upgrade
.PHONY: pixi-upgrade

## Export pixi environment config to conda .yml
pixi2conda:
	@echo "+ $@"
	@pixi workspace export conda-environment --environment notebooks > environment.yml
.PHONY: pixi2conda

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
