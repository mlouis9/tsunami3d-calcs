from direct_pert_calcs import Cases

cases = Cases('case_parameters.yml', 'tsunami-3d-k6')
cases.run_cases('sphere_template.inp')