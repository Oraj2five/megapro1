// Firebase v9 compat module declarations
declare module "firebase/app" {
  export const initializeApp: any;
  export const getApps: any;
  export type FirebaseApp = any;
}

declare module "firebase/auth" {
  export type User = any;
  export const getAuth: any;
  export const signInWithEmailAndPassword: any;
  export const createUserWithEmailAndPassword: any;
  export const GoogleAuthProvider: any;
  export const signInWithPopup: any;
  export const signOut: any;
  export const onAuthStateChanged: any;
  export const updateProfile: any;
}

declare module "firebase/firestore" {
  export type Firestore = any;
  export const getFirestore: any;
  export const doc: any;
  export const setDoc: any;
  export const getDoc: any;
  export const updateDoc: any;
  export const serverTimestamp: any;
  export const collection: any;
  export const getDocs: any;
}

declare module "firebase/storage" {
  export type FirebaseStorage = any;
  export const getStorage: any;
  export const ref: any;
  export const uploadBytes: any;
  export const getDownloadURL: any;
}

