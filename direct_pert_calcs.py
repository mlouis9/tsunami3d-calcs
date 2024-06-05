from pyparsing import *
import yaml
from string import Template
import subprocess
import numpy as np
import os
from abc import ABC, abstractmethod
from uncertainties import ufloat

class Cases:
    def __init__(self, case_parameters_yml, case_sequence):
        self.case_sequence = case_sequence
        self.cases = self._read_case_parameters(case_parameters_yml)

    def _read_case_parameters(self, case_parameters_yml: str):
        """
        Parameters
        ----------
        - case_parameters_yml : str
            Path to the case parameters yaml file
        - template_file : str
            Path to the TSUNAMI 1D template file

        Returns
        -------
        - cases : dict
        """
        # Load the cases
        with open(case_parameters_yml) as f:
            case_parameters = yaml.safe_load(f)

        reflector_materials = case_parameters['reflector_materials']
        heu_material = case_parameters['heu_material']
        cases = case_parameters['cases']

        parsed_cases = []

        for case in cases:
            INCHES_TO_CM = 2.54
            
            model_number = case['model_number']
            heu_radius = case['heu_radius_cm']
            reflector_material = reflector_materials[case['reflector_material']]
            reflector_thickness = case['reflector_thickness_inches'] * INCHES_TO_CM
            reflector_radius = heu_radius + reflector_thickness

            # Now create a case and add it to the parsed cases list
            this_case = Case(
                model_number, 
                heu_material,
                heu_radius, 
                reflector_material, 
                reflector_radius, 
                f'sphere_model_{model_number}.inp',
                self.case_sequence
            )
            parsed_cases.append(this_case)

        return parsed_cases
    
    def run_cases(self, template_file: str):
        for case in self.cases:
            case.create_input_file(template_file)
            case.run_case()

class Case:
    def __init__(self, model_number, heu_material, heu_radius, reflector_material, reflector_radius, case_input, case_sequence='tsunami-1d'):
        """Creates a case from a model number, HEU radius, reflector material, reflector radius, case input file, and
        optionally a case sequence, which defaults to tsunami-1d."""

        self.model_number = model_number
        self.heu_material = heu_material
        self.heu_radius = heu_radius
        self.reflector_material = reflector_material
        self.reflector_radius = reflector_radius
        self.case_input = case_input
        self.case_sequence = case_sequence
        

    def create_input_file(self, template_file: str):
        """
        Parameters
        ----------
        - template_file : str
            Path to the TSUNAMI 1D template file
        - case : dict
            Case dictionary containing the following
        """
        # Read the template file
        with open(template_file) as f:
            template = Template(f.read())

        with open(self.case_input, 'w') as f:
            # Create case input file
            f.write(template.substitute(
                model_number=self.model_number,
                heu_material=self.heu_material,
                heu_radius=self.heu_radius,
                reflector_material=self.reflector_material,
                reflector_radius=self.reflector_radius,
                sequence=self.case_sequence
            ))

    def run_case(self):
        # Run the case and wait for it to finish
        proc = subprocess.Popen(['scalerte', self.case_input])
        proc.wait()

class Output:
    def __init__(self, filename):
        self.filename = filename
        with open(filename) as f:
            self.data = f.read()
 
class Tsunami1DOutput(Output):
    def __init__(self, filename):
        super().__init__(filename)
        self.total_sensitivity_coefficients = self._get_total_sensitivity_coefficients()
        self.keff = self._get_keff()

    def _get_total_sensitivity_coefficients(self):
        # --------------------
        #  Define the parser
        # --------------------

        # Skip lines that are not needed
        unused_lines = SkipTo(Literal('Total Sensitivity Coefficients by Nuclide'))

        # Data row definition
        integer = pyparsing_common.integer
        float_number = pyparsing_common.fnumber
        data_row = Group(integer + Combine(Word(alphas) + '-' + integer) + float_number + float_number)

        # Define the entire block of data to capture
        data_block = OneOrMore(data_row)

        # Define the table and data headers
        header_line = OneOrMore(OneOrMore('-'))
        table_header = Literal("Mixture") + Literal("Nuclide") + Literal("Atom Density") + Literal("Sensitivity")

        # Define the total parser
        total_sensitivity_coefficients_parser = \
            Suppress(
                unused_lines + \
                Literal('Total Sensitivity Coefficients by Nuclide') + \
                header_line + LineEnd() + \
                table_header + \
                header_line + LineEnd()
            ) + \
            data_block
        
        parsed_data = total_sensitivity_coefficients_parser.parseString(self.data)

        # Now parse the list of row lists into a dictionary of dictionaries, where the outer dictionary is keyed by
        # mixture and the inner dictionary is keyed by nuclide

        total_sensitivity_coefficients = {}
        for row in parsed_data:
            mixture = str(row[0])
            nuclide = row[1]
            atom_density = row[2]
            sensitivity = row[3]

            if mixture not in total_sensitivity_coefficients.keys():
                total_sensitivity_coefficients[mixture] = {}
            
            total_sensitivity_coefficients[mixture][nuclide] = {
                'atom_density': atom_density,
                'sensitivity': sensitivity
            }

        return total_sensitivity_coefficients
    
    def _get_keff(self):
        "Gets keff from the forward calculation"

        # Skip lines that are not needed
        unused_lines = SkipTo(Literal("Forward Calculation: k-eff"))

        keff = Suppress(unused_lines + Literal("Forward Calculation: k-eff") + "=") + pyparsing_common.fnumber
        return keff.parseString(self.data)[0]

class Tsunami1DCOutput(Output):
    def __init__(self, filename):
        super().__init__(filename)
        self.keff = self._get_keff()
        
    def _get_keff(self):
        # Skip lines that are not needed
        unused_lines = SkipTo(Literal("lambda"))

        keff = Suppress(unused_lines + Literal("lambda")) + pyparsing_common.fnumber
        return keff.parseString(self.data)[0]
    
class Tsunami3DCEOutput(Output):
    "Parse output from TSUNAMI-3D CE cases"
    def __init__(self, filename):
        super().__init__(filename)
        self.keff = self._get_keff()
        self.total_sensitivity_coefficients = self._get_total_sensitivity_coefficients()
        
    def _get_total_sensitivity_coefficients(self):
        # Skip lines that are not needed
        unused_lines = SkipTo(Literal('Total Sensitivity Coefficients by Nuclide'))

        # Data row definition
        integer = pyparsing_common.integer
        float_number = pyparsing_common.fnumber
        data_row = Group(integer + Combine(Word(alphas) + '-' + integer) + float_number + float_number \
                         + Suppress(Literal("+/-")) + float_number + Suppress("(") + float_number + Suppress("%)"))

        # Define the entire block of data to capture
        data_block = OneOrMore(data_row)

        # Define the table and data headers
        header_line = OneOrMore(OneOrMore('-'))
        table_header = Literal("Mixture") + Literal("Nuclide") + Literal("Atom Density") + Literal("Sensitivity") + \
                        Literal("Std. Dev.") + Literal("% Std. Dev.")

        # Define the total parser
        total_sensitivity_coefficients_parser = \
            Suppress(
                unused_lines + \
                Literal('Total Sensitivity Coefficients by Nuclide') + \
                header_line + LineEnd() + \
                table_header + \
                header_line + LineEnd()
            ) + \
            data_block
        parsed_data = total_sensitivity_coefficients_parser.parseString(self.data)

        # Now parse the list of row lists into a dictionary of dictionaries, where the outer dictionary is keyed by
        # mixture and the inner dictionary is keyed by nuclide

        total_sensitivity_coefficients = {}
        for row in parsed_data:
            mixture = str(row[0])
            nuclide = row[1]
            atom_density = row[2]
            sensitivity = row[3]
            sensitivity_std_dev = row[4]

            if mixture not in total_sensitivity_coefficients.keys():
                total_sensitivity_coefficients[mixture] = {}
            
            total_sensitivity_coefficients[mixture][nuclide] = {
                'atom_density': atom_density,
                'sensitivity': ufloat(sensitivity, sensitivity_std_dev)
            }

        return total_sensitivity_coefficients

    def _get_keff(self):
        # Skip lines that are not needed
        unused_lines = SkipTo(Literal("best estimate system k-eff"))

        keff_and_uncertainty = Suppress(unused_lines + Literal("best estimate system k-eff")) + \
               pyparsing_common.fnumber + Suppress(Literal("+ or -")) + pyparsing_common.fnumber
        result = keff_and_uncertainty.parseString(self.data)
        return ufloat(result[0], result[1])

class KenoVIOutput(Output):
    def __init__(self, filename):
        super().__init__(filename)
        self.keff, self.keff_uncertainty = self._get_keff_and_uncertainty()
    
    def _get_keff_and_uncertainty(self):
        # Skip lines that are not needed
        unused_lines = SkipTo(Literal("best estimate system k-eff"))

        keff_and_uncertainty = Suppress(unused_lines + Literal("best estimate system k-eff")) + \
               pyparsing_common.fnumber + Suppress(Literal("+ or -")) + pyparsing_common.fnumber
        return keff_and_uncertainty.parseString(self.data)

class DirectPerturbationCalculation(ABC):
    """Class to perform direct perturbation calculations for a given case. This class will read the nominal output file and use
    the case parameters to perform the direct perturbation calculations. The results will be stored in the 
    nominal_total_sensitivity_coefficients and output as a .csv file."""
    def __init__(self, case: Case, template_file, overwrite_output=False):
        self.case = case
        self.overwrite_output = overwrite_output
        self.nominal_output = None
        self.nominal_total_sensitivity_coefficients = None
        self.nominal_keff = None
        self.template_file = template_file

    def run_calculation(self):
        nominal_output = self._get_tsunami_output(case.case_input.replace('.inp', '.out'))
        self.nominal_total_sensitivity_coefficients = nominal_output.total_sensitivity_coefficients
        self.nominal_keff = nominal_output.keff

        # Run the direct perturbation calculation for each nuclide and mixture in the nominal_total_sensitivity_coefficients dict
        print(f"Running direct perturbation calculation for model {(self.case).model_number}...")
        for mixture in self.nominal_total_sensitivity_coefficients.keys():
            for nuclide in self.nominal_total_sensitivity_coefficients[mixture].keys():
                self._run_and_process_direct_perturbation_calculation(mixture, nuclide)

        # Finally output sensitivities to .csv
        self._output_sensitivities()
        pass

    @abstractmethod
    def _get_tsunami_output(self, output_path: str):
        pass

    @abstractmethod
    def _get_direct_output(self, output_path: str):
        pass

    @abstractmethod
    def _direct_calculation_sequence(self) -> str:
        """Name of the calculation sequence for doing the direct perturbation calculation, e.g. tsunami-1dc"""
        pass

    @abstractmethod
    def _run_and_process_direct_perturbation_calculation(self):
        pass

    @abstractmethod
    def _calculate_sensitivity_edits(self, perturbation_outputs, rho_deltas, rho_nom, nom_keff):
        """Calculate sensitivity edits, e.g. sensitivity coefficient, and for monte carlo cases the
        uncertainty, etc."""
        pass

    @abstractmethod
    def _output_sensitivities(self):
        pass

    def _run_direct_perturbation_calculation(self, nuclide: str, mixture: str, num_density_points=2):
        """Do the direct perturbation calculation for a given nuclide and mixture, with a default density discretization of 2
        points"""
        if num_density_points != 2:
            raise ValueError('Only two density points are currently supported')

        # -----------------------------------------
        # First calculate the density perturbation
        # -----------------------------------------
        # Read the nominal output
        S_tot = self.nominal_total_sensitivity_coefficients[mixture][nuclide]['sensitivity']
        rho_nom = self.nominal_total_sensitivity_coefficients[mixture][nuclide]['atom_density']
        nom_keff = self.nominal_keff

        # The 0.005 (0.5% for a near critical system) is our target change in keff (i.e. Δk), which provides a ≥ 10 σ difference
        # k+ - k- for σ/k ≤ 0.001
        fractional_rho_delta = ((0.005 / nom_keff) / S_tot).nominal_value
        rho_delta = fractional_rho_delta*rho_nom
        rho_deltas = np.linspace(-rho_delta, rho_delta, num_density_points)

        perturbation_outputs = []
        for rho_index, rho_delta in enumerate(rho_deltas):
            # -----------------
            # Perturb materials
            # -----------------
            # Default to the nominal materials
            perturbed_heu_material = (self.case).heu_material
            perturbed_reflector_material = (self.case).reflector_material

            if mixture == '1':
                perturbed_heu_material = self._material_perturbation(mixture, nuclide, rho_delta)
            elif mixture == '10':
                perturbed_reflector_material = self._material_perturbation(mixture, nuclide, rho_delta)
            else:
                raise ValueError('Mixture must be either 1 or 10')

            # Create a new case
            new_case = Case(
                (self.case).model_number,
                perturbed_heu_material,
                (self.case).heu_radius,
                perturbed_reflector_material,
                (self.case).reflector_radius,
                f'sphere_model_{self.case.model_number}/perturbed_{nuclide}_{mixture}_{rho_index+1}.inp',
                case_sequence=self._direct_calculation_sequence()
            )

            # First, make sphere_model_{model_number} directory if it doesn't exist already
            if not os.path.exists(f'sphere_model_{self.case.model_number}'):
                os.makedirs(f'sphere_model_{self.case.model_number}')

            # Create the input file
            new_case.create_input_file(self.template_file)

            print(f"""  Perturbing {nuclide} density with Δρ={rho_delta:1.4E} for mixture {mixture}...""")
            # Run the case if the output file doesn't already exist
            output_path = new_case.case_input.replace('.inp', '.out')
            if not os.path.exists(output_path) or self.overwrite_output:
                new_case.run_case()

            # Read the output
            perturbation_output = self._get_direct_output(output_path)

            perturbation_outputs.append(perturbation_output)

        # ------------------------------------------------
        # Calculate Edits (Sensitivity coefficients, etc.)
        # ------------------------------------------------

        return self._calculate_sensitivity_edits(perturbation_outputs, rho_deltas, rho_nom, nom_keff)

    def _material_perturbation(self, mixture: str, nuclide: str, rho_delta: float):
        # Get nominal material composition
        if mixture == '1':
            material = (self.case).heu_material
        elif mixture == '10':
            material = (self.case).reflector_material
        else:
            raise ValueError('Mixture must be either 1 or 10')
        
        # Perturb the density of the selected nuclide
        for line in material.split('\n'):
            if nuclide in line:
                # Get the nuclide density
                material_card = Suppress(SkipTo(pyparsing_common.fnumber + Literal("end"))) + pyparsing_common.fnumber
                nuclide_density = material_card.parseString(line)[0]
                new_nuclide_density = nuclide_density + rho_delta

                # Note the exponent has a leading zero by default when formatting using scientific notation, so it must be
                # removed
                perturbed_line = line.replace(f"{nuclide_density:1.4E}".replace("E+0", "E+").replace("E-0", "E-"), \
                                              f"{new_nuclide_density:1.6E}")

                material = material.replace(line, perturbed_line)
                break

        return material

class Tsunami1D_DPCalculation(DirectPerturbationCalculation):
    def __init__(self, case: Case, template_file, overwrite_output=False):
        super().__init__(case, template_file, overwrite_output)

    def _get_tsunami_output(self, output_path: str):
        return Tsunami1DOutput(output_path)
    
    def _get_direct_output(self, output_path: str):
        return Tsunami1DCOutput(output_path)
    
    def _direct_calculation_sequence(self) -> str:
        return 'tsunami-1dc'
    
    def _run_and_process_direct_perturbation_calculation(self, mixture, nuclide):
        dp_sensitivity, keffs, delta_rhos = self._run_direct_perturbation_calculation(nuclide, mixture)

        # Now store the dp sensitivity in the nominal_total_sensitivity_coefficients dict
        this_row = self.nominal_total_sensitivity_coefficients[mixture][nuclide]
        this_row.update({"direct_perturbation_sensitivity": dp_sensitivity})
        this_row.update({"relative_error": (this_row['sensitivity'] - dp_sensitivity)/this_row['sensitivity']*100})
        
        # Parameters for the dp calculation
        this_row.update({"keff+": keffs[1]})
        this_row.update({"keff-": keffs[0]})
        this_row.update({"delta_rho+": delta_rhos[1]})
        this_row.update({"delta_rho-": delta_rhos[0]})

    def _calculate_sensitivity_edits(self, perturbation_outputs, rho_deltas, rho_nom, nom_keff):
        keffs = [output.keff for output in perturbation_outputs]
        sensitivity_coefficient = rho_nom/nom_keff*(keffs[1] - keffs[0]) / (rho_deltas[1] - rho_deltas[0])

        return sensitivity_coefficient, (keffs[0], keffs[1]), (rho_deltas[0], rho_deltas[1])
    
    def _output_sensitivities(self):
        header = ['Mixture', 'Nuclide', 'Atom Density', 'Tsunami 1D Sensitivity', 'Direct Perturbation Sensitivity', \
                  'Relative Error (%)', 'k-eff+', 'k-eff-', 'Δρ+', 'Δρ-']
        rows = []
        for mixture in self.nominal_total_sensitivity_coefficients.keys():
            for nuclide in self.nominal_total_sensitivity_coefficients[mixture].keys():
                properties = self.nominal_total_sensitivity_coefficients[mixture][nuclide]
                row = [mixture, nuclide] + [properties['atom_density'], properties['sensitivity'], \
                                            properties['direct_perturbation_sensitivity'], properties['relative_error'], \
                                            properties['keff+'], properties['keff-'], properties['delta_rho+'], properties['delta_rho-']]
                rows.append(row)

        with open(f'sphere_model_{(self.case).model_number}_comparison.csv', 'w') as f:
            f.write(','.join(header) + '\n')
            for row in rows:
                # Format the row and write it to the file
                f.write(','.join(map(str, row[:2])) + f", {row[2]:1.4E}, {row[3]:1.4E}, {row[4]:1.4E}, {row[5]:2.1f}, " \
                        + ','.join(map(str, row[6:8])) + f", {row[8]:1.4E}, {row[9]:1.4E}" + '\n')

class Tsunami3DCE_DPCalculation(DirectPerturbationCalculation):
    def __init__(self, case: Case, template_file, overwrite_output=False):
        super().__init__(case, template_file, overwrite_output)

    def _get_tsunami_output(self, output_path: str):
        return Tsunami3DCEOutput(output_path)
    
    def _get_direct_output(self, output_path: str):
        return KenoVIOutput(output_path)
    
    def _direct_calculation_sequence(self) -> str:
        return 'csas6'
    
    def _run_and_process_direct_perturbation_calculation(self, mixture, nuclide):
        dp_sensitivity, keffs, delta_rhos = self._run_direct_perturbation_calculation(nuclide, mixture)

        # Now store the dp sensitivity in the nominal_total_sensitivity_coefficients dict
        this_row = self.nominal_total_sensitivity_coefficients[mixture][nuclide]
        this_row.update({"direct_perturbation_sensitivity": dp_sensitivity})
        this_row.update({"relative_error": (this_row['sensitivity'] - dp_sensitivity)/this_row['sensitivity']*100})
        
        # Parameters for the dp calculation
        this_row.update({"keff+": keffs[1]})
        this_row.update({"keff-": keffs[0]})
        this_row.update({"delta_rho+": delta_rhos[1]})
        this_row.update({"delta_rho-": delta_rhos[0]})

    def _calculate_sensitivity_edits(self, perturbation_outputs, rho_deltas, rho_nom, nom_keff):
        keffs = [ufloat(output.keff, output.keff_uncertainty) for output in perturbation_outputs]
        sensitivity_coefficient = rho_nom/nom_keff*(keffs[1] - keffs[0]) / (rho_deltas[1] - rho_deltas[0])
        # manual_calculation = ((keffs[1].std_dev**2 + keffs[0].std_dev**2)/(keffs[1].nominal_value - keffs[0].nominal_value)**2 + nom_keff.std_dev**2/nom_keff.nominal_value**2)**(1/2)*rho_nom/(rho_deltas[1] - rho_deltas[0])
        # print(sensitivity_coefficient, manual_calculation)

        return sensitivity_coefficient, (keffs[0], keffs[1]), (rho_deltas[0], rho_deltas[1])
    
    def _output_sensitivities(self):
        header = ['Mixture', 'Nuclide', 'Atom Density', 'Tsunami 1D Sensitivity', 'Direct Perturbation Sensitivity', \
                  'Relative Error (%)', 'k-eff+', 'k-eff-', 'Δρ+', 'Δρ-']
        rows = []
        for mixture in self.nominal_total_sensitivity_coefficients.keys():
            for nuclide in self.nominal_total_sensitivity_coefficients[mixture].keys():
                properties = self.nominal_total_sensitivity_coefficients[mixture][nuclide]
                row = [mixture, nuclide] + [properties['atom_density'], properties['sensitivity'], \
                                            properties['direct_perturbation_sensitivity'], properties['relative_error'], \
                                            properties['keff+'], properties['keff-'], properties['delta_rho+'], properties['delta_rho-']]
                rows.append(row)

        with open(f'sphere_model_{(self.case).model_number}_comparison.csv', 'w') as f:
            f.write(','.join(header) + '\n')
            for row in rows:
                # Format the row and write it to the file
                f.write(','.join(map(str, row[:2])) + f", {row[2]:1.4E}, {row[3]:1.4E}, {row[4]:1.4E}, {row[5]:2.1f}, " \
                        + ','.join(map(str, row[6:8])) + f", {row[8]:1.4E}, {row[9]:1.4E}" + '\n')

if __name__ == '__main__':
    # First parse the cases
    cases = Cases('case_parameters.yml', case_sequence='tsunami-3d-keno6').cases

    # Assume the nominal cases have already been run, now do the direct perturbation calculations
    for case in cases:
        # Now perform a direct perturbation calculation
        calculation = Tsunami3DCE_DPCalculation(case, 'sphere_template_dp.inp')
        calculation.run_calculation()