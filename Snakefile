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

# these rules are all about big comparisons using all of the eca they have
# hard coded N values which is less than ideal
rule summarize_eca:
    input:
        expand(
            "data/eca/attractors/rule_{rule}/approx_attr_{rule}_" + N + ".csv",
            rule=config['unique_eca_rules'])
    output:
        'plots/eca/transient_hist_by_class_{N}.png',
        "data/eca/attractors/eca_{N}_summary.csv"
    shell:
        'python scripts/summarize_eca.py {wildcards.N}'

rule eca_canalization:
    input:
        'data/eca/attractors/eca_{N}_summary.csv',
        'data/eca/canalization_df.csv'
    output:
        'plots/eca/mean_transient_by_ke_{N}.png',
        'plots/eca/mean_transient_by_ks_{N}.png',
        'plots/eca/coef_transient_by_ke_{N}.png'
    shell:
        'python scripts/plot_eca_canalization.py {wildcard.N}'

rule eca_imin:
    input:
        'data/eca/attractors/eca_{N}_summary.csv',
        'data/eca/imin_df.csv'
    output:
        'plots/eca/mean_transient_by_excess_synergy_imin_{N}.png',
        'plots/eca/mean_transient_by_shared_imin_{N}.png',
        'plots/eca/mean_transient_by_synergy_imin_{N}.png',
        'plots/eca/mean_transient_by_unq_center_imin_{N}.png',
        'plots/eca/mean_transient_by_unq_side_imin_{N}.png'
    shell:
        'python scripts/plot_eca_info.py {wildcard.N} imin'

rule eca_pm:
    input:
        'data/eca/attractors/eca_{N}_summary.csv',
        'data/eca/pm_df.csv'
    output:
        'plots/eca/mean_transient_by_excess_synergy_pm_{N}.png',
        'plots/eca/mean_transient_by_shared_pm_{N}.png',
        'plots/eca/mean_transient_by_synergy_pm_{N}.png',
        'plots/eca/mean_transient_by_unq_center_pm_{N}.png',
        'plots/eca/mean_transient_by_unq_side_pm_{N}.png'
    shell:
        'python scripts/plot_eca_info.py {wildcards.N} pm'
