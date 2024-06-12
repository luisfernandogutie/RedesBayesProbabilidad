import tkinter as tk
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

class BayesianNetworkApp:
    def __init__(self, master):
        self.master = master
        master.title("Red Bayesiana en Medicina")

        # Etiquetas y entradas para ingresar datos
        self.label_enfermedad = tk.Label(master, text="Nombre de la enfermedad:", font=("Arial", 12))
        self.label_enfermedad.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.enfermedad_entry = tk.Entry(master, font=("Arial", 12))
        self.enfermedad_entry.grid(row=0, column=1, padx=10, pady=5)

        self.label_sintoma1 = tk.Label(master, text="Síntoma 1 (Temperatura alta):", font=("Arial", 12))
        self.label_sintoma1.grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.sintoma1_entry = tk.Entry(master, font=("Arial", 12))
        self.sintoma1_entry.grid(row=1, column=1, padx=10, pady=5)

        self.label_sintoma2 = tk.Label(master, text="Síntoma 2 (Dolor de cabeza):", font=("Arial", 12))
        self.label_sintoma2.grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.sintoma2_entry = tk.Entry(master, font=("Arial", 12))
        self.sintoma2_entry.grid(row=2, column=1, padx=10, pady=5)

        self.label_sintoma3 = tk.Label(master, text="Síntoma 3 (Fatiga):", font=("Arial", 12))
        self.label_sintoma3.grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.sintoma3_entry = tk.Entry(master, font=("Arial", 12))
        self.sintoma3_entry.grid(row=3, column=1, padx=10, pady=5)

        # Botón para calcular la probabilidad
        self.calcular_button = tk.Button(master, text="Calcular Probabilidad", command=self.calcular_probabilidad, font=("Arial", 12))
        self.calcular_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

        # Etiqueta para mostrar el resultado
        self.resultado_label = tk.Label(master, text="", font=("Arial", 12))
        self.resultado_label.grid(row=5, column=0, columnspan=2, padx=10, pady=5)

        # Etiqueta para mostrar errores
        self.error_label = tk.Label(master, text="", fg="red", font=("Arial", 12))
        self.error_label.grid(row=6, column=0, columnspan=2, padx=10, pady=5)

    def validar_entrada(self, entrada):
        """Valida que la entrada sea un número entre 0 y 1."""
        try:
            valor = float(entrada)
            if 0 <= valor <= 1:
                return True
            else:
                return False
        except ValueError:
            return False

    def calcular_probabilidad(self):
        """Realiza el cálculo de la probabilidad de la enfermedad dada la evidencia de los síntomas."""
        nombre_enfermedad = self.enfermedad_entry.get()
        sintoma1 = self.sintoma1_entry.get()
        sintoma2 = self.sintoma2_entry.get()
        sintoma3 = self.sintoma3_entry.get()

        # Validación de la entrada
        if not self.validar_entrada(sintoma1) or not self.validar_entrada(sintoma2) or not self.validar_entrada(sintoma3):
            self.error_label.config(text="Los síntomas deben ser números entre 0 y 1")
            return

        sintoma1 = float(sintoma1)
        sintoma2 = float(sintoma2)
        sintoma3 = float(sintoma3)

        # Definición del modelo de la red bayesiana
        model = BayesianModel([(nombre_enfermedad, 'Temperatura alta'), (nombre_enfermedad, 'Dolor de cabeza'), (nombre_enfermedad, 'Fatiga')])

        # Definición de las distribuciones de probabilidad condicional
        cpd_enfermedad = TabularCPD(variable=nombre_enfermedad, variable_card=2, values=[[0.01], [0.99]])
        cpd_sintoma1 = TabularCPD(variable='Temperatura alta', variable_card=2,
                                   values=[[0.9, 0.2],
                                           [0.1, 0.8]],
                                   evidence=[nombre_enfermedad],
                                   evidence_card=[2])
        cpd_sintoma2 = TabularCPD(variable='Dolor de cabeza', variable_card=2,
                                   values=[[0.7, 0.3],
                                           [0.3, 0.7]],
                                   evidence=[nombre_enfermedad],
                                   evidence_card=[2])
        cpd_sintoma3 = TabularCPD(variable='Fatiga', variable_card=2,
                                   values=[[0.8, 0.2],
                                           [0.2, 0.8]],
                                   evidence=[nombre_enfermedad],
                                   evidence_card=[2])

        # Agregar las distribuciones de probabilidad al modelo
        model.add_cpds(cpd_enfermedad, cpd_sintoma1, cpd_sintoma2, cpd_sintoma3)

        # Realizar la inferencia
        inference = VariableElimination(model)
        resultado = inference.query(variables=[nombre_enfermedad],
                                    evidence={'Temperatura alta': sintoma1, 'Dolor de cabeza': sintoma2, 'Fatiga': sintoma3})

        # Mostrar el resultado
        self.resultado_label.config(text=f"Probabilidad de {nombre_enfermedad}: {resultado.values[1]:.4f}")
        self.error_label.config(text="")  # Limpiar mensaje de error si existe

def main():
    root = tk.Tk()
    app = BayesianNetworkApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
