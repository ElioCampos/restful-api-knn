// RESTful API que llama al algoritmo knn implementado en la carpeta "source/knn"

package main

import (
	"source/knn"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"github.com/gorilla/mux"
)

type KNNRequest struct {
	Url        string `json:"url"`
	Cols       int    `json:"cols"`
	KNeighbors       int    `json:"neighbors"`
}

type KNNResult struct {
	Trained 	map[string]int 	`json:"trained"`
	Tested 		map[string]int 	`json:"tested"`
	Correct    	int 			`json:"correct"`
	Prediction 	int 			`json:"prediction"`
	Accuracy   	float64 		`json:"accuracy"`
}

var KNNRequests KNNRequest
var KNNResults KNNResult

func homePage(writer http.ResponseWriter, httpRequest *http.Request) {
	fmt.Fprintf(writer, "RESTful API - KNN")
	fmt.Println("Detalle de endpoint: homePage")
}

func manejaRutas() {
	enrutador := mux.NewRouter().StrictSlash(true)
	enrutador.HandleFunc("/", homePage)
	enrutador.HandleFunc("/requests", retornaRequests).Methods("GET")
	enrutador.HandleFunc("/request", crearRequest).Methods("POST")
	enrutador.HandleFunc("/results", retornaResultados).Methods("GET")
	log.Fatal(http.ListenAndServe(":8000", enrutador))
}

func retornaRequests(writer http.ResponseWriter, httpRequest *http.Request){
	writer.Header().Set("Access-Control-Allow-Origin", "*")
    	fmt.Println("Detalle de endpoint: retornaRequests")
    	json.NewEncoder(writer).Encode(KNNRequests)
}

func crearRequest(writer http.ResponseWriter, httpRequest *http.Request) {
	writer.Header().Set("Access-Control-Allow-Origin", "*")
   	contenidoReq, _ := ioutil.ReadAll(httpRequest.Body)
	var nuevoRequest KNNRequest 
   	json.Unmarshal(contenidoReq, &nuevoRequest)
	KNNRequests = nuevoRequest
	prediccionesCorrectas, prediccionesTotales, precisionFinal, entrenadas, testeadas := knn.AlgoritmoKNN(nuevoRequest.Url, nuevoRequest.Cols, nuevoRequest.KNeighbors)
	fmt.Println(prediccionesCorrectas, prediccionesTotales, precisionFinal, entrenadas, testeadas)
	KNNResults = KNNResult { Trained: entrenadas, Tested: testeadas, Correct: prediccionesCorrectas,
		Prediction: prediccionesTotales, Accuracy: precisionFinal }
}

func retornaResultados(writer http.ResponseWriter, httpRequest *http.Request){
	writer.Header().Set("Access-Control-Allow-Origin", "*")
    	fmt.Println("Detalle de endpoint: retornaResultados")
   	json.NewEncoder(writer).Encode(KNNResults)
}

func main() {
	manejaRutas()
}
