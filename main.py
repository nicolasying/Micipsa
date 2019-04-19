# test script
import importlib
import sys

from lib import utils
env = utils.Env('./', 'fr')

core_number = -2

base_model_1 = utils.Model('rms', env)
base_model_2 = utils.Model('rms-wrate', env)
base_model_3 = utils.Model('rms-wrate-cwrate', env)
sig_model = utils.Model('rms-wrate-cwrate-sig200', env)
asn_model = utils.Model('rms-wrate-cwrate-asn200', env)
# model = asn_model
# model.config_model()
# model.generate_individual_results(core_number=core_number, verbose=True)
sim_model = utils.Model('rms-wrate-cwrate-sim100', env)
mix_model = utils.Model('rms-wrate-cwrate-mix200', env)

for model in [sig_model, asn_model, sim_model, mix_model, base_model_3, base_model_2, base_model_1]:
    model.config_model()
    model.print_model_config()
    # model.generate_design_matrices()
    model.generate_individual_results(core_number=core_number, verbose=True)
    model.generate_group_results()


comp1 = utils.ModelComparison(comparison_name='rms-wrate-cwrate-sig-asn', environment=env)
comp2 = utils.ModelComparison(comparison_name='rms-wrate-cwrate-sim-asn', environment=env)
comp3 = utils.ModelComparison(comparison_name='rms-wrate-cwrate-sim-sig-mix', environment=env)

for comp in [comp1, comp2, comp3]:
    comp.config_comp()
    comp.print_comp_config()
    comp.generate_individual_results()
    comp.generate_group_results()
    comp.generate_report()

# del env
# for model in [base_model_1, base_model_2, base_model_3, sig_model, asn_model, sim_model, mix_model]:
#     del model
# importlib.reload(utils)