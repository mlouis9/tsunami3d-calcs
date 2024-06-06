from direct_pert_calcs import Cases
import sys

if __name__ == '__main__':
    # First parse the cases
    cases = Cases('case_parameters.yml', 'tsunami-3d-k6').cases

    if len(sys.argv) != 1:
        # If a specific case is requested, run that (useful for doing multiple runs in parallel)
        case = cases[int(sys.argv[1])]
        case.create_input_file('sphere_template.inp')
        case.run_case()
    else:
        # RUn all inputs
        for case in cases:
            case.create_input_file('sphere_template.inp')
            case.run_case()