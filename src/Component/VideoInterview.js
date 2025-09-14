import React, { useRef, useEffect, useState } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as blazeface from '@tensorflow-models/blazeface';
import '@tensorflow/tfjs';
import styles from './VideoInterview.module.css';

const suspiciousItems = ['cell phone', 'book', 'laptop', 'keyboard', 'remote'];

const VideoInterview = () => {
  const videoRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [faceDetected, setFaceDetected] = useState(false);
  const [eventLogs, setEventLogs] = useState([]);
  const [newEventIndex, setNewEventIndex] = useState(null);
  const [candidateName, setCandidateName] = useState('');
  const [startTime, setStartTime] = useState(Date.now());

  const lastFaceCheckTime = useRef(Date.now());
  const lastMultipleFacesTime = useRef(0);
  const lastItemDetectionTime = useRef(0);
  const lastLookAwayTime = useRef(0);
  const detectionInterval = useRef(null);

  // --- Start Video ---
  useEffect(() => {
    const startVideo = async () => {
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        videoRef.current.srcObject = mediaStream;
        setStream(mediaStream);
        setStartTime(Date.now());
      } catch (error) {
        console.error('Error accessing webcam:', error);
      }
    };
    startVideo();

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);  // Added 'stream' as dependency

  // --- Detection Utilities ---
  const checkLookingAway = (face, videoWidth) => {
    const faceCenterX = face.topLeft[0] + (face.bottomRight[0] - face.topLeft[0]) / 2;
    const normalizedX = faceCenterX / videoWidth;
    return normalizedX < 0.35 || normalizedX > 0.65;
  };

  const checkEyesClosed = (face, videoHeight) => {
    const faceHeight = face.bottomRight[1] - face.topLeft[1];
    const ratio = faceHeight / videoHeight;
    return ratio < 0.1;
  };

  // --- Main Detection Loop ---
  useEffect(() => {
    let cocoModel, faceModel;
    let isMounted = true;

    const loadModelsAndRun = async () => {
      cocoModel = await cocoSsd.load();
      faceModel = await blazeface.load();
      console.log('COCO-SSD and BlazeFace models loaded');

      detectionInterval.current = setInterval(async () => {
        if (!isMounted || !videoRef.current || videoRef.current.readyState < 2) return;

        const [cocoPredictions, facePredictions] = await Promise.all([
          cocoModel.detect(videoRef.current),
          faceModel.estimateFaces(videoRef.current, false)
        ]);

        const currentTime = Date.now();
        const timestamp = new Date().toISOString();

        const addEvent = (eventText) => {
          setEventLogs(prev => {
            const updated = [...prev, { timestamp, event: eventText }];
            setNewEventIndex(updated.length - 1);
            return updated;
          });
        };

        // Face Detection
        if (facePredictions.length === 0 && currentTime - lastFaceCheckTime.current > 5000) {
          addEvent('No face detected for more than 5 seconds');
          lastFaceCheckTime.current = currentTime;
          setFaceDetected(false);
        } else if (facePredictions.length >= 1) {
          lastFaceCheckTime.current = currentTime;
          setFaceDetected(true);
        }

        // Multiple Faces
        if (facePredictions.length > 1 && currentTime - lastMultipleFacesTime.current > 5000) {
          addEvent(`Multiple users detected (${facePredictions.length})`);
          lastMultipleFacesTime.current = currentTime;
        }

        // Suspicious Items
        if (currentTime - lastItemDetectionTime.current > 2000) {
          cocoPredictions.forEach(pred => {
            if (suspiciousItems.includes(pred.class)) {
              addEvent(`Suspicious Item Detected: ${pred.class}`);
            }
          });
          lastItemDetectionTime.current = currentTime;
        }

        // Look Away & Eyes Closed
        if (facePredictions.length === 1) {
          const face = facePredictions[0];
          if (checkLookingAway(face, videoRef.current.videoWidth) &&
              currentTime - lastLookAwayTime.current > 5000) {
            addEvent('User looking away');
            lastLookAwayTime.current = currentTime;
          }
          if (checkEyesClosed(face, videoRef.current.videoHeight)) {
            addEvent('User eyes closed');
          }
        } else if (facePredictions.length > 1) {
          const lookingAwayCount = facePredictions.filter(face => checkLookingAway(face, videoRef.current.videoWidth)).length;
          const eyesClosedCount = facePredictions.filter(face => checkEyesClosed(face, videoRef.current.videoHeight)).length;

          if (lookingAwayCount > 0) addEvent(`${lookingAwayCount} users looking away`);
          if (eyesClosedCount > 0) addEvent(`${eyesClosedCount} users eyes closed`);
        }
      }, 1000);
    };

    loadModelsAndRun();

    return () => {
      isMounted = false;
      if (detectionInterval.current) clearInterval(detectionInterval.current);
    };
  }, []);

  // --- Export CSV ---
  const exportLogsAsCSV = () => {
    const interviewDuration = ((Date.now() - startTime) / 1000).toFixed(0) + ' sec';
    const focusLostEvents = eventLogs.filter(log => log.event.includes('No face detected')).length;
    const multipleFacesEvents = eventLogs.filter(log => log.event.includes('Multiple users detected')).length;
    const suspiciousItemEvents = eventLogs.filter(log => log.event.includes('Suspicious Item Detected')).length;
    const eyesClosedEvents = eventLogs.filter(log => log.event.includes('eyes closed')).length;
    const lookingAwayEvents = eventLogs.filter(log => log.event.includes('looking away')).length;

    const totalDeductions = focusLostEvents*5 + multipleFacesEvents*5 + suspiciousItemEvents*10 + eyesClosedEvents*10 + lookingAwayEvents*5;
    const integrityScore = Math.max(0, 100 - totalDeductions);

    const summary = [
      ['Candidate Name', candidateName || 'Unknown'],
      ['Interview Duration', interviewDuration],
      ['Focus Lost Events', focusLostEvents],
      ['Multiple Users Events', multipleFacesEvents],
      ['Suspicious Item Events', suspiciousItemEvents],
      ['Eyes Closed Events', eyesClosedEvents],
      ['Looking Away Events', lookingAwayEvents],
      ['Final Integrity Score', integrityScore]
    ];

    const logLines = eventLogs.map(log => `${log.timestamp},${log.event}`);
    const csvContent =
      'data:text/csv;charset=utf-8,' +
      summary.map(row => row.join(',')).join('\n') +
      '\n\nEvent Logs:\nTimestamp,Event\n' +
      logLines.join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'proctoring_report.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className={styles.container}>
      <h2>Candidate Interview</h2>

      <input
        type="text"
        value={candidateName}
        onChange={e => setCandidateName(e.target.value)}
        placeholder="Enter candidate name"
        className={styles.inputField}
      />

      <div className={styles.interviewWrapper}>
        <div className={styles.videoWrapper}>
          <video ref={videoRef} autoPlay playsInline className={styles.video} />
          <div className={`${styles.status} ${faceDetected ? styles.statusGreen : styles.statusRed}`}>
            {faceDetected ? 'Face detected ✅' : 'No face detected ❌'}
          </div>
          <button className={styles.button} onClick={exportLogsAsCSV}>
            Export Proctoring Report (CSV)
          </button>
        </div>

        <div className={styles.eventLogs}>
          <h3>Event Logs</h3>
          <ul>
            {eventLogs.map((log, idx) => (
              <li
                key={idx}
                className={idx === newEventIndex ? styles.newEvent : ''}
                onAnimationEnd={() => setNewEventIndex(null)}
              >
                [{log.timestamp}] → {log.event}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default VideoInterview;
