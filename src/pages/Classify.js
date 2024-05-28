import React, { Component, Fragment } from 'react';
import {
  Alert, Button, Collapse, Container, Form, Spinner, ListGroup, Tabs, Tab
} from 'react-bootstrap';
import { FaCamera, FaChevronDown, FaChevronRight } from 'react-icons/fa';
import { openDB } from 'idb';
import Cropper  from 'react-cropper';
import * as tf from '@tensorflow/tfjs';
import LoadButton from '../components/LoadButton';
import { MODEL_CLASSES } from '../model/classes';
import config from '../config';
import './Classify.css';
import 'cropperjs/dist/cropper.css';
import labels from "../model/labels.json";

//CNN part
const MODEL_PATH = '/model/model.json';

const IMAGE_SIZE_X = 224;
const IMAGE_SIZE_Y = 224;

const CNN_MODEL_SIZE = 224;

const CANVAS_SIZE_X = 224;
const CANVAS_SIZE_Y = 224;

const YOLO_CANVAS_SIZE = 640;
const CAMERA_SIZE_X = 640;
const CAMERA_SIZE_Y = 640;

const TOPK_PREDICTIONS = 2;

const INDEXEDDB_DB = 'tensorflowjs';
const INDEXEDDB_STORE = 'model_info_store';
const INDEXEDDB_KEY = 'web-mobilenetv2-model';

// Yolo Part
const numClass = labels.length;

const YOLO_MODEL_PATH = '/yolov8_model/model.json';

const INDEXEDDB_YOLO_DB = 'yolo_model';
const INDEXEDDB_YOLO_STORE = 'yolo_model_info_store';
const INDEXEDDB_YOLO_KEY = 'web_yolo_model';

/**
 * @extends React.Component
 */
export default class Classify extends Component {

  constructor(props) {
    super(props);

    this.webcam = null;
    this.model = null;
    this.modelLastUpdated = null;
    this.yoloModel = null;
    this.yoloModelLastUpdated = null;

    this.state = {
      modelLoaded: false,
      yoloModelLoaded: false,
      filename: '',
      isModelLoading: false,
      isYoloModelLoading: false,
      isClassifying: false,
      isDetecting: false,
      scores_data: null,
      predictions: [],
      photoSettingsOpen: true,
      modelUpdateAvailable: false,
      yoloModelUpdateAvailable: false,
      showModelUpdateAlert: false,
      showYoloModelUpdateAlert: false,
      showModelUpdateSuccess: false,
      showYoloModelUpdateSuccess: false,
      isDownloadingModel: false,
      isDownloadingYoloModel: false
    };
  }

  async componentDidMount() {
    if (('indexedDB' in window)) {
      try {
        this.model = await tf.loadLayersModel('indexeddb://' + INDEXEDDB_KEY);

        try {
          const db = await openDB(INDEXEDDB_DB, 1, );
          const item = await db.transaction(INDEXEDDB_STORE)
                               .objectStore(INDEXEDDB_STORE)
                               .get(INDEXEDDB_KEY);
          const dateSaved = new Date(item.modelArtifactsInfo.dateSaved);
          await this.getModelInfo();
          console.log(this.modelLastUpdated);
          if (!this.modelLastUpdated  || dateSaved >= new Date(this.modelLastUpdated).getTime()) {
            console.log('Using saved model');
          }
          else {
            this.setState({
              modelUpdateAvailable: true,
              showModelUpdateAlert: true,
            });
          }

        }
        catch (error) {
          console.warn(error);
          console.warn('Could not retrieve when model was saved.');
        }

      }
      catch (error) {
        console.log('Not found in IndexedDB. Loading and saving...');
        console.log(error);
        this.model = await tf.loadLayersModel(MODEL_PATH);
        await this.model.save('indexeddb://' + INDEXEDDB_KEY);
      }
      try {
        this.yoloModel = await tf.loadGraphModel('indexeddb://' + INDEXEDDB_YOLO_KEY);
        try{
            const db = await openDB(INDEXEDDB_YOLO_DB, 1 , );
            const item = await db.transaction(INDEXEDDB_YOLO_STORE)
                                .objectStore(INDEXEDDB_YOLO_STORE)
                                .get(INDEXEDDB_YOLO_KEY);
            const dateSaved = new Date(item.modelArtifactsInfo.dateSaved);
            await this.getModelInfo();
            console.log(this.yoloModelLastUpdated);
            if (!this.yoloModelLastUpdated || dateSaved >= new Date(this.yoloModelLastUpdated).getTime()){
                console.log('Using saved yolo model');
            }
            else {
                this.setState({
                    yoloModelUpdateAvailable: true,
                    showYoloModelUpdateAlert: true,
                });
            }
        }
        catch (error){
            console.warn(error);
            console.warn('Could not retrieve when yolo model was saved.');
        }
      }
      catch(error){
        console.log('Not found in IndexedDB. Loading and saving...');
        console.log(error);
        this.model = await tf.loadGraphModel(YOLO_MODEL_PATH);
        await this.model.save('indexeddb://' + INDEXEDDB_YOLO_KEY);
      }
    }
    else {
      console.warn('IndexedDB not supported.');
      this.model = await tf.loadLayersModel(MODEL_PATH);
      this.yoloModel = await tf.loadGraphModel(YOLO_MODEL_PATH);
    }

    this.setState({ modelLoaded: true });
    this.setState({yoloModelLoaded: true});
    this.initWebcam();

    let prediction = tf.tidy(() => this.model.predict(tf.zeros([1, CNN_MODEL_SIZE, CNN_MODEL_SIZE, 3])));
    prediction.dispose();

    const dummyInput = tf.ones(this.yoloModel.inputs[0].shape);
    const warmUpResults = this.yoloModel.execute(dummyInput);

    tf.dispose([warmUpResults,dummyInput])


  }

  async componentWillUnmount() {
    if (this.webcam) {
      this.webcam.stop();
    }

    try {
      this.model.dispose();
    }
    catch (e) {
    }
  }

  initWebcam = async () => {
    try {
      this.webcam = await tf.data.webcam(
        this.refs.webcam,
        {resizeWidth: CAMERA_SIZE_X, resizeHeight: CAMERA_SIZE_Y, facingMode: 'environment'}
      );
    }
    catch (e) {
      this.refs.noWebcam.style.display = 'block';
    }
  }

  startWebcam = async () => {
    if (this.webcam) {
      this.webcam.start();
    }
  }

  stopWebcam = async () => {
    if (this.webcam) {
      this.webcam.stop();
    }
  }

  getModelInfo = async () => {
    await fetch(`${config.API_ENDPOINT}/model_info`, {
      method: 'GET',
    })
    .then(async (response) => {
      await response.json().then((data) => {
        this.modelLastUpdated = data.last_updated;
      })
      .catch((err) => {
        console.log('Unable to get parse model info.');
      });
    })
    .catch((err) => {
      console.log('Unable to get model info');
    });
    await fetch(`${config.API_ENDPOINT}/yolo_model_info`, {
      method: 'GET',
  })
  .then(async (response) => {
      await response.json().then((data) => {
        this.yoloModelLastUpdated = data.last_updated;
      })
      .catch((err) => {
        console.log('Unable to get parse model info.');
      });
  })
  .catch((err) => {
      console.log('Unable to get yolo model info');
  });
  }

  updateModel = async () => {
    console.log('Updating the model: ' + INDEXEDDB_KEY);
    this.setState({ isDownloadingModel: true });
    this.model = await tf.loadLayersModel(MODEL_PATH);
    await this.model.save('indexeddb://' + INDEXEDDB_KEY);
    console.log('Updating the yolo model: ' + INDEXEDDB_YOLO_KEY);
    this.setState({ isDownloadingYoloModel: true });
    this.model = await tf.loadLayersModel(YOLO_MODEL_PATH);
    await this.model.save('indexeddb://' + INDEXEDDB_YOLO_KEY);
    this.setState({
      isDownloadingModel: false,
      modelUpdateAvailable: false,
      showModelUpdateAlert: false,
      showModelUpdateSuccess: true,
      isDownloadingYoloModel: false,
      yoloModelUpdateAvailable: false,
      showYoloModelUpdateAlert: false,
      showYoloModelUpdateSuccess: true
    });
  }

  classifyLocalImage = async () => {
    this.setState({ isClassifying: true });

    const [modelWidth, modelHeight] = this.yoloModel.inputs[0].shape.slice(1,3);

    const croppedCanvas = this.refs.cropper.getCroppedCanvas();
    // const image = tf.tidy( () => tf.browser.fromPixels(croppedCanvas).toFloat());

    const input = tf.tidy(() => {
      const img = tf.browser.fromPixels(croppedCanvas);

      const [h, w] = img.shape.slice(0, 2); 
      const maxSize = Math.max(w, h); 
      const imgPadded = img.pad([
        [0, maxSize - h], 
        [0, maxSize - w], 
        [0, 0],
      ]);

      return tf.image
                .resizeBilinear(imgPadded, [modelWidth, modelHeight]) 
                .div(255.0) 
                .expandDims(0); 
              
    });
      
    tf.engine().startScope(); 

    const res = this.yoloModel.execute(input); 
    const transRes = res.transpose([0, 2, 1]); 
    const boxes = tf.tidy(() => {

      const w = transRes.slice([0, 0, 2], [-1, -1, 1]); 
      const h = transRes.slice([0, 0, 3], [-1, -1, 1]); 

      const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); 
      const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); 
      return tf
          .concat(
          [
          y1,
          x1,
          tf.add(y1, h), 
          tf.add(x1, w), 
          ],
          2
          )
          .squeeze();
    }); 
            
    const [scores, classes] = tf.tidy(() => {
        
        const rawScores = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0); 
        return [rawScores.max(1), rawScores.argMax(1)];
    }); 
            
    const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2); 
  
    const boxes_data = boxes.gather(nms, 0).dataSync(); 
    const scores_data = scores.gather(nms, 0).dataSync(); 

    tf.engine().endScope(); 
    this.setState({
      scores_data : scores_data

    });
    
    const context = this.refs.canvas.getContext('2d');
    const ratioX = 640 / croppedCanvas.width;
    const ratioY = 640 / croppedCanvas.height;
    const ratio = Math.min(ratioX, ratioY);
    context.clearRect(0, 0, CANVAS_SIZE_X, CANVAS_SIZE_Y);

    const img_x = (boxes_data[1])/ratio
    const img_y = (boxes_data[0])/ratio
    const img_w = (boxes_data[2] - boxes_data[0])/ratio
    const img_h = (boxes_data[3] - boxes_data[1])/ratio

    context.drawImage(croppedCanvas, img_x , img_y, img_w, img_h,0,0,224,224);


    
    const detectedCanvas = this.refs.canvas.getContext('2d');
    const canvasImageData = detectedCanvas.getImageData(0, 0, 224, 224);
    const image = tf.tidy( () => tf.browser.fromPixels(canvasImageData));

    const imageData = await this.processImage(image);
    const resizedImage = tf.image.resizeBilinear(imageData, [IMAGE_SIZE_X, IMAGE_SIZE_Y]);

    const logits = this.model.predict(resizedImage);
    const probabilities = await logits.data();
    const preds = await this.getTopKClasses(probabilities, TOPK_PREDICTIONS);

    this.setState({
      predictions: preds,
      isClassifying: false,
      photoSettingsOpen: !this.state.photoSettingsOpen
    });

    image.dispose();
    imageData.dispose();
    resizedImage.dispose();
    logits.dispose();
  }

  classifyWebcamImage = async () => {
    this.setState({ isClassifying: true,
    scores_data: 1 });
    const [modelWidth, modelHeight] = this.yoloModel.inputs[0].shape.slice(1,3);

    const imageCapture = await this.webcam.capture();

    const tensorData = tf.tidy(() => imageCapture.toFloat().div(255));
    await tf.browser.toPixels(tensorData, this.refs.canvas);

    const contextImg = this.refs.canvas.getContext('2d');
    const contextImgR = contextImg.getImageData(0,0,640,640);

    const input = tf.tidy(() => {
      const img = tf.browser.fromPixels(contextImgR);

      const [h, w] = img.shape.slice(0, 2); 
      const maxSize = Math.max(w, h); 
      const imgPadded = img.pad([
        [0, maxSize - h], 
        [0, maxSize - w], 
        [0, 0],
      ]);

      return tf.image
                .resizeBilinear(imgPadded, [modelWidth, modelHeight]) 
                .div(255.0) 
                .expandDims(0); 

      });

      tf.engine().startScope();
      const res = this.yoloModel.execute(input); 
      const transRes = res.transpose([0, 2, 1]); 
      const boxes = tf.tidy(() => {

      const w = transRes.slice([0, 0, 2], [-1, -1, 1]); 
      const h = transRes.slice([0, 0, 3], [-1, -1, 1]); 

      const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); 
      const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); 
      return tf
            .concat(
            [
            y1,
            x1,
            tf.add(y1, h), 
            tf.add(x1, w), 
            ],
            2
            )
            .squeeze();
      }); 

      const [scores, classes] = tf.tidy(() => {

        const rawScores = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0); 
        return [rawScores.max(1), rawScores.argMax(1)];
      }); 
      const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2); 

      const boxes_data = boxes.gather(nms, 0).dataSync(); 
      const scores_data = scores.gather(nms, 0).dataSync(); 
      tf.engine().endScope(); 

      this.setState({
        scores_data: scores_data,
        isDetecting: false,
        photoSettingsOpen: !this.state.photoSettingsOpen
      });

      
      const context = this.refs.canvas.getContext('2d');
      const ratioX = YOLO_CANVAS_SIZE / contextImgR.width;
      const ratioY = YOLO_CANVAS_SIZE / contextImgR.height;
      const ratio = Math.min(ratioX, ratioY);
      context.clearRect(0, 0, YOLO_CANVAS_SIZE, YOLO_CANVAS_SIZE);

      const img_x = ((boxes_data[1])/ratio)|0;
      const img_y = ((boxes_data[0])/ratio)|0;
      const img_w = ((boxes_data[2] - boxes_data[0])/ratio)|0;
      const img_h = ((boxes_data[3] - boxes_data[1])/ratio)|0;

      const tensorDataR = tf.tidy(() => imageCapture.toFloat().div(255).slice([img_y,img_x,0],[img_h,img_w,-1]));
      const upscaleTensorDataR = tf.tidy(()=>tf.image.resizeBilinear(tensorDataR,[224,224]));
      await tf.browser.toPixels(upscaleTensorDataR, this.refs.canvas);

 
      
      const detectedCanvas = this.refs.canvas.getContext('2d');
      const canvasImageData = detectedCanvas.getImageData(0, 0, 224, 224);
      const image = tf.tidy( () => tf.browser.fromPixels(canvasImageData));

      const imageData = await this.processImage(image);
      const resizedImage = tf.image.resizeBilinear(imageData, [IMAGE_SIZE_X, IMAGE_SIZE_Y]);

      const logits = this.model.predict(resizedImage);
      const probabilities = await logits.data();
      const preds = await this.getTopKClasses(probabilities, TOPK_PREDICTIONS);

      this.setState({
        predictions: preds,
        isClassifying: false,
        photoSettingsOpen: !this.state.photoSettingsOpen
      });

      resizedImage.dispose();
      imageCapture.dispose();
      imageData.dispose();
      logits.dispose();
      tensorData.dispose();
  }

  processImage = async (image) => {
    return tf.tidy(() => image.expandDims(0).toFloat().div(127).sub(1));
  }

  /**
   * @param logits 
   * @param topK 
   */
  getTopKClasses = async (values, topK) => {
    const valuesAndIndices = [];
    for (let i = 0; i < values.length; i++) {
      valuesAndIndices.push({value: values[i], index: i});
    }
    valuesAndIndices.sort((a, b) => {
      return b.value - a.value;
    });
    const topkValues = new Float32Array(topK);
    const topkIndices = new Int32Array(topK);
    for (let i = 0; i < topK; i++) {
      topkValues[i] = valuesAndIndices[i].value;
      topkIndices[i] = valuesAndIndices[i].index;
    }

    const topClassesAndProbs = [];
    for (let i = 0; i < topkIndices.length; i++) {
      topClassesAndProbs.push({
        className: MODEL_CLASSES[topkIndices[i]],
        probability: (topkValues[i] * 100).toFixed(2)
      });
    }
    return topClassesAndProbs;
  }

  handlePanelClick = event => {
    this.setState({ photoSettingsOpen: !this.state.photoSettingsOpen });
  }

  handleFileChange = event => {
    if (event.target.files && event.target.files.length > 0) {
      this.setState({
        file: URL.createObjectURL(event.target.files[0]),
        filename: event.target.files[0].name
      });
    }
  }

  handleTabSelect = activeKey => {
    switch(activeKey) {
      case 'camera':
        this.startWebcam();
        break;
      case 'localfile':
        this.setState({filename: null, file: null});
        this.stopWebcam();
        break;
      default:
    }
  }

  render() {
    return (
      <div className="Classify container">

      { !this.state.modelLoaded &&
        <Fragment>
          <Spinner animation="border" role="status">
            <span className="sr-only">Loading...</span>
          </Spinner>
          {' '}<span className="loading-model-text">Loading Models</span>
        </Fragment>
      }

      { this.state.modelLoaded &&
        <Fragment>
        <Button
          onClick={this.handlePanelClick}
          className="classify-panel-header"
          aria-controls="photo-selection-pane"
          aria-expanded={this.state.photoSettingsOpen}
          >
          Take or Select a Photo to Classify
            <span className='panel-arrow'>
            { this.state.photoSettingsOpen
              ? <FaChevronDown />
              : <FaChevronRight />
            }
            </span>
          </Button>
          <Collapse in={this.state.photoSettingsOpen}>
            <div id="photo-selection-pane">
            { this.state.modelUpdateAvailable && this.state.showModelUpdateAlert &&
                <Container>
                  <Alert
                    variant="info"
                    show={this.state.modelUpdateAvailable && this.state.showModelUpdateAlert}
                    onClose={() => this.setState({ showModelUpdateAlert: false})}
                    dismissible>
                      An update for the <strong>{this.state.modelType}</strong> model is available.
                      <div className="d-flex justify-content-center pt-1">
                        {!this.state.isDownloadingModel &&
                          <Button onClick={this.updateModel}
                                  variant="outline-info">
                            Update
                          </Button>
                        }
                        {this.state.isDownloadingModel &&
                          <div>
                            <Spinner animation="border" role="status" size="sm">
                              <span className="sr-only">Downloading...</span>
                            </Spinner>
                            {' '}<strong>Downloading...</strong>
                          </div>
                        }
                      </div>
                  </Alert>
                </Container>
              }
              {this.state.showModelUpdateSuccess &&
                <Container>
                  <Alert variant="success"
                         onClose={() => this.setState({ showModelUpdateSuccess: false})}
                         dismissible>
                    The <strong>{this.state.modelType}</strong> model has been updated!
                  </Alert>
                </Container>
              }
            <Tabs defaultActiveKey="camera" id="input-options" onSelect={this.handleTabSelect}
                  className="justify-content-center">
              <Tab eventKey="camera" title="Take Photo">
                <div id="no-webcam" ref="noWebcam">
                  <span className="camera-icon"><FaCamera /></span>
                  No camera found. <br />
                  Please use a device with a camera,give access to camera or upload an image instead.
                </div>
                <div className="webcam-box-outer">
                  <div className="webcam-box-inner">
                    <video ref="webcam" autoPlay playsInline muted id="webcam"
                           width="480" height="480">
                    </video>
                  </div>
                </div>
                <div className="button-container">
                  <LoadButton
                    variant="primary"
                    size="lg"
                    onClick={this.classifyWebcamImage}
                    isLoading={this.state.isClassifying}
                    text="Classify"
                    loadingText="Classifying..."
                  />
                </div>
              </Tab>
              <Tab eventKey="localfile" title="Select Local File">
                <Form.Group controlId="file">
                  <Form.Label>Select Image File</Form.Label><br />
                  <Form.Label className="imagelabel">
                    {this.state.filename ? this.state.filename : 'Browse...'}
                  </Form.Label>
                  <Form.Control
                    onChange={this.handleFileChange}
                    type="file"
                    accept="image/*"
                    className="imagefile" />
                </Form.Group>
                { this.state.file &&
                  <Fragment>
                    <div id="local-image">
                      <Cropper
                        ref="cropper"
                        src={this.state.file}
                        style={{height: 500, width: 281.25}}
                        guides={true}
                        aspectRatio={ 9/ 16}
                        viewMode={2}
                      />
                    </div>
                    <div className="button-container">
                      <LoadButton
                        variant="primary"
                        size="lg"
                        disabled={!this.state.filename}
                        onClick={this.classifyLocalImage}
                        isLoading={this.state.isClassifying}
                        text="Classify"
                        loadingText="Classifying..."
                      />
                    </div>
                  </Fragment>
                }
              </Tab>
            </Tabs>
            </div>
          </Collapse>
          { this.state.scores_data  > 0 &&
            <div className="classification-results">
              <h3>Prediction</h3>
              <canvas ref="canvas" width={CANVAS_SIZE_X} height={CANVAS_SIZE_Y} />
              <br />
              <ListGroup>
              {this.state.predictions.map((category) => {
                  return (
                    <ListGroup.Item key={category.className}>
                      <strong>{category.className}</strong> {category.probability}%</ListGroup.Item>
                  );
              })}
              </ListGroup>
            </div>
          }
          </Fragment>
        }
      </div>
    );
  }
}
