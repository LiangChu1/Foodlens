import React, { useState, useRef } from "react";
import Webcam from "react-webcam";
import urlFile from './output.txt'
import './App.css'
import { Container, Row, Col } from 'reactstrap';
import 'bootstrap/dist/css/bootstrap.css';


const App = () => {
    // Ref for accessing the webcam
    const webcamRef = useRef(null);
    // State variables for image and nutrition info
    const [image, setImage] = useState(null);
    const [imageURL, setImageURL] = useState('');
    const [isCapturing, setIsCapturing] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [label, setLabel] = useState("");
    const [calories, setCalories] = useState(""); 
    const [protein, setProtein] = useState(""); 
    const [fat, setFat] = useState(""); 
    const [carbs, setCarbs] = useState(""); 

     // Headers for API requests
    const myHeaders = new Headers();
    myHeaders.append("Content-Type", "application/json");

    // Function to upload image data to server
    const uploadImage = async (apiUrl, data) => {
        const raw = JSON.stringify({
            "body": {
                "imageData": data
            }
        });

        const requestOptions = {
            method: "POST",
            headers: myHeaders,
            body: raw,
            redirect: "follow"
        };

        const response = await fetch(apiUrl + "/upload", requestOptions);
        const result = await response.json();
        return result.url;
    };

    // Function to classify image and get predicted label
    const classifyImage = async (apiUrl, imagePath) => {
        const raw = JSON.stringify({
            "image_path": imagePath
        });

        const requestOptions = {
            method: "POST",
            headers: myHeaders,
            body: raw,
            redirect: "follow"
        };

        const response = await fetch(apiUrl + "/classify", requestOptions);
        const result = await response.json();
        // Replace underscores with spaces
        let predictedLabel = result.predicted_label.replaceAll("_", " ");

        // Capitalize the first letter
        predictedLabel = predictedLabel.charAt(0).toUpperCase() + predictedLabel.slice(1);
        setLabel(predictedLabel);
        return predictedLabel;
    };

    // Function to get nutrition info based on predicted label
    const getNutrition = async (apiUrl, predictedLabel) => {
        const raw = JSON.stringify({
            "body": {
                "predicted_label": predictedLabel
            }
        });

        const requestOptions = {
            method: "POST",
            headers: myHeaders,
            body: raw,
            redirect: "follow"
        };

        const response = await fetch(apiUrl + "/nutrition", requestOptions);
        const result = await response.json();
        const { nf_calories, nf_total_fat, nf_protein, nf_total_carbohydrate } = result.nutrition_data;

        setCalories(nf_calories);
        setFat(nf_total_fat);
        setCarbs(nf_total_carbohydrate);
        setProtein(nf_protein);
    };

    //calls the functionality above to detect the nutrition info of the meal within the image
    const handleDetectNutitionInfo = async (data) => {
        const response = await fetch(urlFile);
        const apiUrl = await response.text();

        const imageUrl = await uploadImage(apiUrl, data);
        const predictedLabel = await classifyImage(apiUrl, imageUrl);
        await getNutrition(apiUrl, predictedLabel);
    };

    // Function to handle image capture from webcam
   const captureNew = async () => {
       setIsCapturing(true);

       const screenshot = webcamRef.current.getScreenshot();
       if (screenshot) {
           const blob = await fetch(screenshot).then(res => res.blob());
           setImage(blob);
           setImageURL(screenshot);
       } else {
           console.error("Invalid image format. Please capture a PNG or JPEG image.");
       }
       setIsCapturing(false);
   };

    // Function to handle image upload
    const handleUpload = async () => {
        setIsUploading(true);

        try {
            // Convert blob to base64
            const reader = new FileReader();
            reader.onloadend = () => {
                let base64data = reader.result;
                base64data = base64data.replace(/^data:image\/\w+;base64,/, '');
                handleDetectNutitionInfo(base64data);
                setIsUploading(false);
            };
            reader.readAsDataURL(image);
        } catch (error) {
            console.error("Error uploading image:", error);
            setIsUploading(false);
        }
    };

    // Function to handle file change
   const handleFileChange = (e) => {
       const file = e.target.files[0];
       if (file) {
         const imageUrl = URL.createObjectURL(file);
         setImageURL(imageUrl);
         setImage(file);
       } else {
         console.error("Invalid image format. Please capture a PNG or JPEG image.");
       }
   };



    return (
        <div className="App">
            <Container className="Sections" fluid>
                <Row>
                    <div className="bannerHeader">
                        FoodLens
                    </div>
                </Row>
                <Row style={{ display: 'flex', justifyContent: 'center' }}>
                    <div className="center">
                        <img src={require("./Images/logo.png")} alt="Home" />
                    </div>
                </Row>
                <Row style={{ display: 'flex', justifyContent: 'center' }}>
                    <Col className="cameraCol" lg={4} md={4} sm={12} style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center'}}>
                        <div className="leftSide">
                            <Webcam
                                className="cameraStyle"
                                audio={false}
                                screenshotFormat="image/jpeg"
                                ref={webcamRef}
                            />
                            <div className="InteractButtons">
                                <button className="captureButton" onClick={captureNew} disabled={isCapturing || isUploading}>
                                    Capture Camera
                                </button>
                                <button className="uploadButton" onClick={handleUpload} disabled={!image || isUploading}>
                                    Upload New Image
                                </button>
                            </div>
                            <label className="textColor" for="fileInput">Upload File Here</label>
                            <input type="file" id="fileInput" onChange={handleFileChange} accept="image/*" />
                        </div>
                    </Col>
                    <Col className="dataCol" lg={4} md={4} sm={12} style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                        <div className="rightSide">
                            <Container className="foodData" style={{ border: '3px solid black' }}>
                                <div>
                                    <p className="textColor">Food Name: {label}</p>
                                </div>
                                <div>
                                    <p className="textColor">Calories: {calories}</p>
                                </div>
                                <div>
                                    <p className="textColor">Fats: {fat}g</p>
                                </div>
                                <div>
                                    <p className="textColor">Carbohydrates: {carbs}g</p>
                                </div>
                                <div>
                                    <p className="textColor">Protein: {protein}g</p>
                                </div>
                            </Container>
                        </div>
                    </Col>
                    <Col className="imageCol" lg={4} md={4} sm={12} style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center' }}>
                        <Container className="capturedImage" fluid>
                            {isCapturing && <p>Capturing image...</p>}
                            {isUploading && <p>Uploading image...</p>}
                            {
                                image && imageURL ? (
                                    <div>
                                        <img className="capturedImage" src={imageURL} alt="Captured" />
                                    </div>
                                ) : null
                            }
                        </Container>
                    </Col>
                </Row>
            </Container>
        </div >
    );
};


export default App;
