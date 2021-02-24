configfile: 'workflow/config.yml'

### The first group of rules will serve to validate the  approximation
### I'm using to find the periods and transients in larger ECA
### simulations where constructing STGs is impossible
rule eca_stg:
    input:
        "data/eca/eca_equiv_classes.csv"
    output:
        "data/eca/stgs/rule_{rule}.edgelist"
    shell:
        "python scripts/eca_stgs.py {wildcards.rule}"

rule stg_attractors:
    input:
        "data/eca/stgs/rule_{rule}.edgelist"
    output:
        sp="data/eca/attractors/rule_{rule}/stg_population_periods_{rule}.txt",
        st="data/eca/attractors/rule_{rule}/stg_population_transients_{rule}.txt"
    shell:
        'python scripts/stg_attractors.py {wildcards.rule}'

# this one will just do all of the stg calculations for all of the rules
rule all_stg_stats:
    input:
        expand(
            "data/eca/attractors/rule_{rule}/stg_population_periods_{rule}.txt",
            rule=config['unique_eca_rules'])

rule simulation_attractors:
    input:
        "data/eca/eca_equiv_classes.csv"
    output:
        ap="data/eca/attractors/rule_{rule}/approx_16_periods_{rule}.txt",
        at="data/eca/attractors/rule_{rule}/approx_16_transients_{rule}.txt",
        ep="data/eca/attractors/rule_{rule}/exact_16_periods_{rule}.txt",
        et="data/eca/attractors/rule_{rule}/exact_16_transients_{rule}.txt"
    shell:
        "python scripts/sim_eca_verify.py {wildcards.rule}"

rule verify_approximation:
    input:
        st="data/eca/attractors/rule_{rule}/stg_population_transients_{rule}.txt",
        et="data/eca/attractors/rule_{rule}/exact_16_transients_{rule}.txt",
        at="data/eca/attractors/rule_{rule}/approx_16_transients_{rule}.txt"
    output:
        ea_plot='plots/verify_approximation/{rule}_ea_plot.png',
        qq_plot='plots/verify_approximation/{rule}_qq_plot.png',
        qq_plot2='plots/verify_approximation/{rule}_qq_plot_matching.png',
        perm_plot='plots/verify_approximation/{rule}_perm_dist.png'
    shell:
        'python scripts/verify_approx.py {wildcards.rule}'


