// Implementación del algoritmo KNN para ser usado por la RESTful API

package knn

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	rt "runtime"
	"sort"
	sc "strconv"
	"sync"
)

var wait = sync.WaitGroup{}

type Vecino struct {
	etiqueta  string
	distancia float64
}

type Vecinos []Vecino

func (datoVecino Vecinos) Swap(i, j int) {
	datoVecino[i], datoVecino[j] = datoVecino[j], datoVecino[i]
}

func (datoVecino Vecinos) Less(i, j int) bool {
	if datoVecino[i].distancia == datoVecino[j].distancia {
		return datoVecino[i].etiqueta < datoVecino[j].etiqueta
	} else {
		return datoVecino[i].distancia < datoVecino[j].distancia
	}
}

func (datoVecino Vecinos) Len() int {
	return len(datoVecino)
}

func distanciaEuclid(pos1, pos2 []float64) float64 {
	var distancia float64
	for i := range pos1 {
		distancia = distancia + math.Pow(pos1[i]-pos2[i], 2)
	}
	distanciaEuclidiana := math.Sqrt(distancia)
	return distanciaEuclidiana
}

func frec(datosClasificacion []string) map[string]int {
	mapeado := make(map[string]int)
	for _, dato := range datosClasificacion {
		mapeado[dato] = mapeado[dato] + 1
	}
	return mapeado
}

type EstructuraKNN struct {
	k       int
	entrada [][]float64
	salida  []string
}

func (estructuraKNN EstructuraKNN) funcPrediccion(datosTest [][]float64) []string {
	totalPredicciones := make([]string, len(datosTest))
	for valor1, pos1 := range datosTest {
		wait.Add(1)
		go func(totalPredicciones *[]string, valor1 int, estructuraKNN EstructuraKNN, pos1 []float64) {
			totalVecinos := Vecinos{}
			for valor2, pos2 := range estructuraKNN.entrada {
				distanciaEuclid := distanciaEuclid(pos1, pos2)
				vecino := Vecino{estructuraKNN.salida[valor2], distanciaEuclid}
				totalVecinos = append(totalVecinos, vecino)
			}
			sort.Sort(totalVecinos)
			vecinosCercanos := totalVecinos[:estructuraKNN.k]

			var etiquetas []string
			for _, vecino := range vecinosCercanos {
				etiquetas = append(etiquetas, vecino.etiqueta)
			}
			frecuenciaEtiquetas := frec(etiquetas)
			maximo, prediccion := 0, ""
			for i, dato := range frecuenciaEtiquetas {
				if dato > maximo {
					prediccion = i
					maximo = dato
				}
			}
			(*totalPredicciones)[valor1] = prediccion
			wait.Done()
		}(&totalPredicciones, valor1, estructuraKNN, pos1)
	}
	wait.Wait()
	return totalPredicciones
}

func manejaError(err error) {
	if err != nil {
		fmt.Println(err)
	}
}

func leerDataset(url string, cantColumnas int) ([][]float64, []string) {
	dataset, err := http.Get(url)
	manejaError(err)
	defer dataset.Body.Close()
	lectorCSV, err := csv.NewReader(dataset.Body).ReadAll()
	manejaError(err)
	entrada := [][]float64{}
	salida := []string{}

	for i := 1; i < len(lectorCSV); i++ {
		fila := []float64{}
		for _, dato := range lectorCSV[i][:cantColumnas] {
			datoModificado, err := sc.ParseFloat(dato, 64)
			manejaError(err)
			fila = append(fila, datoModificado)
		}
		entrada = append(entrada, fila)
		salida = append(salida, lectorCSV[i][cantColumnas])
	}

	return entrada, salida
}

func divDataset(entrada [][]float64, salida []string, porcEntrVal float64) ([][]float64, []string, [][]float64, []string) {
	var datosEntrenamiento, datosTest [][]float64
	var labelsEntrenamiento, labelsTest []string

	for i := 0; i < len(entrada); i++ {
		limite := rand.Intn(100)
		if limite > int(100*porcEntrVal) {
			datosTest = append(datosTest, entrada[i])
			labelsTest = append(labelsTest, salida[i])
		} else {
			datosEntrenamiento = append(datosEntrenamiento, entrada[i])
			labelsEntrenamiento = append(labelsEntrenamiento, salida[i])
		}
	}

	return datosEntrenamiento, labelsEntrenamiento, datosTest, labelsTest
}

func AlgoritmoKNN(url string, cols int, numVecinos int) (int, int, float64, map[string]int, map[string]int) {
	rt.GOMAXPROCS(1000)
	entrada, salida := leerDataset(url, cols)
	datosEntrenamiento, labelsEntrenamiento, datosTest, labelsTest := divDataset(entrada, salida, 0.5)
	estructuraKNN := EstructuraKNN{
		k:       numVecinos,
		entrada: datosEntrenamiento,
		salida:  labelsEntrenamiento,
	}
	prediccion := estructuraKNN.funcPrediccion(datosTest)
	cantidadCorrecta := 0
	for i := range prediccion {
		if prediccion[i] == labelsTest[i] {
			cantidadCorrecta = cantidadCorrecta + 1
		}
	}
	precision := float64(cantidadCorrecta) / float64(len(prediccion)) * 100
	return cantidadCorrecta, len(prediccion), math.Round(precision*100) / 100, frec(labelsEntrenamiento), frec(labelsTest)
}
