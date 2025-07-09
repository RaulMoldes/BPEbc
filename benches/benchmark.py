import subprocess
import time
import sys
import matplotlib.pyplot as plt

# Configuraciones del benchmark
binary_path = "./bpe"  # Ruta al ejecutable compilado
corpus_path = "corpus.txt"  # Ruta al corpus
num_merges = "100"  # Número de fusiones (puedes cambiarlo)
thread_counts = [1, 2, 4, 8, 12, 16, 32, 64, 128, 256]  # Número de hilos a probar
repeats = 3  # Número de veces que se repite cada medición

print("Benchmarking BPE parallel performance...\n")
avg_times = []
for threads in thread_counts:
    total_time = 0.0
    print(f"Testing with {threads} threads:")

    for i in range(repeats):
        start = time.time()

        # Ejecuta el programa con el número de hilos como argumento
        result = subprocess.run([binary_path, corpus_path, str(threads)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        end = time.time()
        elapsed = end - start
        total_time += elapsed
        print(f"  Run {i+1}: {elapsed:.3f} s")

    avg_time = total_time / repeats
    avg_times.append(avg_time)
    print(f"Average time with {threads} threads: {avg_time:.3f} s\n")


# Calcular speedup relativo al tiempo con 1 hilo
baseline = avg_times[0]
speedups = [baseline / t for t in avg_times]

# Crear gráfico
plt.figure(figsize=(10, 6))
plt.plot(thread_counts, speedups, marker='o', label="Speedup", color="royalblue")
plt.xlabel("Número de hilos")
plt.ylabel("Speedup")
plt.title("Speedup de BPE con OpenMP")
plt.legend()
plt.grid(True)
plt.xscale("log", base=2)
plt.xticks(thread_counts, thread_counts, rotation=45)
plt.tight_layout()
plt.savefig("bpe_speedup.png")

