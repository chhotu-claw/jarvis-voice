// 3D Face with morph targets, shadows, particles, and atmospheric effects
import * as THREE from 'three';
import { GLTFLoader } from '/static/GLTFLoader.js';
import { MeshoptDecoder } from '/static/meshopt_decoder.module.js';

class AsciiFace {
  constructor(container, opts = {}) {
    this.container = container;
    this.width = opts.width || container.clientWidth || 600;
    this.height = opts.height || container.clientHeight || 500;
    this.mouthOpen = 0;
    this.targetMouthOpen = 0;
    this.state = 'idle';
    this.time = 0;
    this.headMesh = null;
    this.morphDict = {};
    this.influences = null;

    this.stateColors = {
      idle: '#00ffc8', speaking: '#00ff88', thinking: '#ff9f1c',
      listening: '#ff3366', wake: '#00ffc8', transcribing: '#8b5cf6',
    };

    this._initScene();
    this._initBackground();
    this._initParticles();
    this._loadFace();
  }

  _initScene() {
    this.scene = new THREE.Scene();
    // Animated gradient background via shader
    const bgGeo = new THREE.PlaneGeometry(2, 2);
    const bgMat = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        resolution: { value: new THREE.Vector2(this.width, this.height) },
      },
      vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position, 1.0); }`,
      fragmentShader: `
        uniform float time;
        uniform vec2 resolution;
        varying vec2 vUv;
        void main() {
          vec2 uv = vUv;
          
          // Slow-moving dark gradients
          float n1 = sin(uv.x * 3.0 + time * 0.12) * sin(uv.y * 2.0 - time * 0.08);
          float n2 = sin(uv.x * 1.5 - time * 0.1 + 2.0) * sin(uv.y * 3.5 + time * 0.06);
          float n3 = sin(length(uv - 0.5) * 4.0 - time * 0.15);
          
          // Visible color shifts over dark base
          vec3 col1 = vec3(0.02, 0.05, 0.12); // blue
          vec3 col2 = vec3(0.06, 0.02, 0.10); // purple
          vec3 col3 = vec3(0.02, 0.08, 0.08); // teal
          
          vec3 color = col1 + col2 * n1 * 0.6 + col3 * n2 * 0.5;
          
          // Vignette
          float vig = 1.0 - length(uv - 0.5) * 0.6;
          color *= vig;
          
          // Moving caustic pattern
          float caustic = sin(uv.x * 8.0 + time * 0.2 + n1 * 2.0) * sin(uv.y * 6.0 - time * 0.15 + n2 * 2.0);
          color += vec3(0.0, 0.04, 0.03) * max(0.0, caustic);
          
          gl_FragColor = vec4(color, 0.85);
        }
      `,
      depthWrite: false,
      depthTest: false,
    });
    this.bgMesh = new THREE.Mesh(bgGeo, bgMat);
    this.bgScene = new THREE.Scene();
    this.bgCamera = new THREE.Camera();
    this.bgScene.add(this.bgMesh);
    this.scene.background = null;
    this.scene.fog = new THREE.FogExp2(0x020206, 0.15);

    this.camera = new THREE.PerspectiveCamera(30, this.width / this.height, 0.1, 50);
    this.camera.position.set(0, 0, 3.5);
    this.camera.lookAt(0, 0, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.renderer.domElement.style.display = 'block';
    this.container.appendChild(this.renderer.domElement);

    // === Lighting ===

    // Ambient — very dim blue
    this.scene.add(new THREE.AmbientLight(0x0a0a20, 0.3));

    // Key light — cool white, casts shadows
    this.keyLight = new THREE.DirectionalLight(0xddeeff, 2.0);
    this.keyLight.position.set(2, 3, 4);
    this.keyLight.castShadow = true;
    this.keyLight.shadow.mapSize.set(1024, 1024);
    this.keyLight.shadow.camera.near = 0.5;
    this.keyLight.shadow.camera.far = 15;
    this.keyLight.shadow.camera.left = -2;
    this.keyLight.shadow.camera.right = 2;
    this.keyLight.shadow.camera.top = 2;
    this.keyLight.shadow.camera.bottom = -2;
    this.keyLight.shadow.bias = -0.002;
    this.keyLight.shadow.radius = 4;
    this.scene.add(this.keyLight);

    // Rim light — cyan, behind-left (state-reactive color)
    this.rimLight = new THREE.PointLight(0x00ffc8, 4, 10);
    this.rimLight.position.set(-2.5, 1, -2);
    this.scene.add(this.rimLight);

    // Rim light 2 — purple, behind-right
    this.rimLight2 = new THREE.PointLight(0x6622cc, 2.5, 8);
    this.rimLight2.position.set(2, 0.5, -1.5);
    this.scene.add(this.rimLight2);

    // Under light — subtle eerie uplight
    this.underLight = new THREE.PointLight(0x002244, 1.5, 5);
    this.underLight.position.set(0, -2, 1);
    this.scene.add(this.underLight);

    // Spot from above — dramatic top-down cone
    this.topSpot = new THREE.SpotLight(0x00ffc8, 2.5, 10, Math.PI * 0.2, 0.6, 1);
    this.topSpot.position.set(0, 4, 1.5);
    this.topSpot.target.position.set(0, 0, 0);
    this.topSpot.castShadow = true;
    this.topSpot.shadow.mapSize.set(512, 512);
    this.scene.add(this.topSpot);
    this.scene.add(this.topSpot.target);
  }

  _initBackground() {
    // Minimal ground — just shadow catcher, invisible
    const groundGeo = new THREE.PlaneGeometry(10, 10);
    const groundMat = new THREE.ShadowMaterial({ opacity: 0.4 });
    this.ground = new THREE.Mesh(groundGeo, groundMat);
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.position.y = -1.8;
    this.ground.receiveShadow = true;
    this.scene.add(this.ground);
  }

  _initParticles() {
    // Sparse, slow-drifting micro particles — like dust in a dark room
    const count = 250;
    const positions = new Float32Array(count * 3);
    this.particleSpeeds = new Float32Array(count);
    this.particlePhases = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 8;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 6;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 5 - 1;
      this.particleSpeeds[i] = 0.01 + Math.random() * 0.04;
      this.particlePhases[i] = Math.random() * Math.PI * 2;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const mat = new THREE.PointsMaterial({
      color: 0x44ffcc, size: 0.04, transparent: true, opacity: 0.6,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
    });

    this.particles = new THREE.Points(geo, mat);
    this.scene.add(this.particles);
  }

  _loadFace() {
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);

    const dummyKTX2 = {
      detectSupport: () => dummyKTX2,
      loadTextureImage: (source, texture, onLoad) => { onLoad(new THREE.Texture()); return texture; },
      load: (url, onLoad) => { onLoad(new THREE.Texture()); },
      parse: () => {},
    };
    loader.setKTX2Loader(dummyKTX2);

    loader.load('/static/facecap.glb', (gltf) => {
      const model = gltf.scene;

      model.traverse((child) => {
        if (child.isMesh) {
          if (child.morphTargetDictionary && Object.keys(child.morphTargetDictionary).length > 10) {
            this.headMesh = child;
            this.morphDict = child.morphTargetDictionary;
            this.influences = child.morphTargetInfluences;
          }

          child.material = new THREE.MeshStandardMaterial({
            color: 0xc0c8d4,
            roughness: 0.55,
            metalness: 0.25,
            flatShading: true,
            envMapIntensity: 0.5,
          });
          child.castShadow = true;
          child.receiveShadow = true;

          child.geometry = child.geometry.toNonIndexed();
          child.geometry.computeVertexNormals();

          // Block animation data
          const pos = child.geometry.attributes.position;
          child.userData.origPositions = new Float32Array(pos.array);
          const faceCount = pos.count / 3;
          const facePhases = new Float32Array(faceCount);
          const faceSpeeds = new Float32Array(faceCount);
          const faceNormals = new Float32Array(faceCount * 3);
          for (let f = 0; f < faceCount; f++) {
            facePhases[f] = Math.random() * Math.PI * 2;
            faceSpeeds[f] = 0.5 + Math.random() * 1.5;
            const ni = f * 9;
            const norms = child.geometry.attributes.normal.array;
            faceNormals[f * 3] = norms[ni];
            faceNormals[f * 3 + 1] = norms[ni + 1];
            faceNormals[f * 3 + 2] = norms[ni + 2];
          }
          child.userData.facePhases = facePhases;
          child.userData.faceSpeeds = faceSpeeds;
          child.userData.faceNormals = faceNormals;
        }
      });

      const box = new THREE.Box3().setFromObject(model);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const scale = 2.6 / Math.max(size.x, size.y, size.z);
      model.scale.setScalar(scale);
      const scaledCenter = center.multiplyScalar(scale);
      model.position.set(-scaledCenter.x, -scaledCenter.y + 0.3, -scaledCenter.z);

      this.faceGroup = model;
      this.scene.add(model);
      this._animate();

      console.log('🎭 Face loaded:', Object.keys(this.morphDict).length, 'blend shapes');
    }, undefined, (err) => console.error('Face load failed:', err));
  }

  _getMorphIndex(name) {
    return this.morphDict[name] ?? this.morphDict['blendShape1.' + name] ?? -1;
  }

  _setMorph(name, value) {
    const idx = this._getMorphIndex(name);
    if (idx >= 0 && this.influences) this.influences[idx] = value;
  }

  setState(state) {
    this.state = state || 'idle';
    const col = new THREE.Color(this.stateColors[this.state] || this.stateColors.idle);
    if (this.rimLight) this.rimLight.color = col;
    if (this.topSpot) this.topSpot.color = col;
  }

  setMouthOpen(v) {
    this.targetMouthOpen = Math.max(0, Math.min(1, v));
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.time += 0.016;
    const t = this.time;

    // Smooth mouth
    this.mouthOpen += (this.targetMouthOpen - this.mouthOpen) * 0.35;

    // === Morph targets ===
    if (this.influences) {
      this._setMorph('jawOpen', this.mouthOpen * 0.35);

      if (this.state === 'speaking' && this.mouthOpen > 0.05) {
        const m = this.mouthOpen;
        this._setMorph('mouthFunnel', Math.sin(t * 6) * 0.12 * m);
        this._setMorph('mouthPucker', Math.max(0, Math.sin(t * 4.5 + 1) * 0.1 * m));
        this._setMorph('mouthStretchLeft', Math.max(0, Math.sin(t * 5.2) * 0.08 * m));
        this._setMorph('mouthStretchRight', Math.max(0, Math.sin(t * 5.2) * 0.08 * m));
        this._setMorph('mouthLeft', Math.sin(t * 3.7) * 0.05 * m);
        this._setMorph('mouthRight', -Math.sin(t * 3.7) * 0.05 * m);
        this._setMorph('mouthShrugUpper', Math.max(0, Math.sin(t * 7) * 0.06 * m));
      } else {
        ['mouthFunnel', 'mouthPucker', 'mouthStretchLeft', 'mouthStretchRight',
         'mouthLeft', 'mouthRight', 'mouthShrugUpper'].forEach(n => this._setMorph(n, 0));
      }

      // Blink
      const blinkPhase = t % 4;
      if (blinkPhase > 3.7 && blinkPhase < 3.9) {
        const b = Math.sin((blinkPhase - 3.7) / 0.2 * Math.PI);
        this._setMorph('eyeBlinkLeft', b);
        this._setMorph('eyeBlinkRight', b);
      } else {
        this._setMorph('eyeBlinkLeft', 0);
        this._setMorph('eyeBlinkRight', 0);
      }

      // Eye look
      this._setMorph('eyeLookInLeft', Math.sin(t * 0.3) * 0.15);
      this._setMorph('eyeLookInRight', -Math.sin(t * 0.3) * 0.15);
      this._setMorph('eyeLookUpLeft', Math.sin(t * 0.4) * 0.1);
      this._setMorph('eyeLookUpRight', Math.sin(t * 0.4) * 0.1);

      // Thinking
      if (this.state === 'thinking') {
        this._setMorph('browInnerUp', 0.4);
        this._setMorph('mouthPressLeft', 0.3);
        this._setMorph('mouthPressRight', 0.3);
      } else {
        this._setMorph('browInnerUp', Math.sin(t * 0.5) * 0.05);
        this._setMorph('mouthPressLeft', 0);
        this._setMorph('mouthPressRight', 0);
      }
    }

    // === Head rotation ===
    if (this.faceGroup) {
      let rotY = Math.sin(t * 0.4) * 0.08 + Math.sin(t * 0.17) * 0.04;
      let rotX = Math.sin(t * 0.3) * 0.03;
      if (this.state === 'speaking') { rotY += Math.sin(t * 1.0) * 0.04; rotX += Math.sin(t * 0.7) * 0.02; }
      else if (this.state === 'thinking') { rotX += Math.sin(t * 0.5) * 0.04 + 0.03; }
      else if (this.state === 'listening') { rotY += 0.05; }
      this.faceGroup.rotation.y = rotY;
      this.faceGroup.rotation.x = rotX;
    }

    // === Block animation ===
    if (this.faceGroup) {
      this.faceGroup.traverse((child) => {
        if (child.isMesh && child.userData.origPositions) {
          const pos = child.geometry.attributes.position.array;
          const orig = child.userData.origPositions;
          const phases = child.userData.facePhases;
          const speeds = child.userData.faceSpeeds;
          const normals = child.userData.faceNormals;
          const faceCount = phases.length;
          for (let f = 0; f < faceCount; f++) {
            let d = Math.sin(t * speeds[f] + phases[f]) * 1.5;
            if (this.state === 'speaking') d += Math.sin(t * speeds[f] * 3 + phases[f]) * 2.5 * this.mouthOpen;
            const nx = normals[f*3], ny = normals[f*3+1], nz = normals[f*3+2];
            for (let v = 0; v < 3; v++) {
              const idx = (f*3+v)*3;
              pos[idx] = orig[idx] + nx*d; pos[idx+1] = orig[idx+1] + ny*d; pos[idx+2] = orig[idx+2] + nz*d;
            }
          }
          child.geometry.attributes.position.needsUpdate = true;
        }
      });
    }

    // === Floating particles — slow drift ===
    if (this.particles) {
      const ppos = this.particles.geometry.attributes.position.array;
      const count = ppos.length / 3;
      for (let i = 0; i < count; i++) {
        ppos[i*3+1] += this.particleSpeeds[i] * 0.016;
        if (ppos[i*3+1] > 3.5) { ppos[i*3+1] = -3; ppos[i*3] = (Math.random()-0.5) * 8; }
        ppos[i*3] += Math.sin(t * 0.3 + this.particlePhases[i]) * 0.003;
        ppos[i*3+2] += Math.cos(t * 0.2 + this.particlePhases[i] * 1.5) * 0.001;
      }
      this.particles.geometry.attributes.position.needsUpdate = true;
    }

    // === Subtle light breathing ===
    if (this.rimLight) this.rimLight.intensity = 3.5 + Math.sin(t * 0.8) * 0.5;
    if (this.rimLight2) this.rimLight2.intensity = 2 + Math.sin(t * 0.6 + 1) * 0.3;
    if (this.underLight) this.underLight.intensity = 1 + this.mouthOpen * 0.8;
    if (this.topSpot) this.topSpot.intensity = 2 + Math.sin(t * 0.7) * 0.3;

    // Render gradient background first, then scene on top
    if (this.bgMesh) {
      this.bgMesh.material.uniforms.time.value = t;
      this.renderer.autoClear = true;
      this.renderer.render(this.bgScene, this.bgCamera);
      // Render main scene without clearing (overlay on bg)
      this.renderer.autoClear = false;
      this.renderer.clearDepth();
      this.renderer.render(this.scene, this.camera);
      this.renderer.autoClear = true;
    } else {
      this.renderer.render(this.scene, this.camera);
    }
  }

  destroy() {
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}

window.AsciiFace = AsciiFace;
