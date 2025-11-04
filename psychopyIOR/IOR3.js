/************* 
 * Ior3 *
 *************/

import { core, data, sound, util, visual, hardware } from './lib/psychojs-2024.2.4.js';
const { PsychoJS } = core;
const { TrialHandler, MultiStairHandler } = data;
const { Scheduler } = util;
//some handy aliases as in the psychopy scripts;
const { abs, sin, cos, PI: pi, sqrt } = Math;
const { round } = util;


// store info about the experiment session:
let expName = 'IOR3';  // from the Builder filename that created this script
let expInfo = {
    'participant': `${util.pad(Number.parseFloat(util.randint(0, 999999)).toFixed(0), 6)}`,
    'session': '001',
};

// Start code blocks for 'Before Experiment'
// Run 'Before Experiment' code from practice_code
prefs.hardware["skipFrameCheck"] = true;

// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([(- 1), (- 1), (- 1)]),
  units: 'height',
  waitBlanking: true,
  backgroundImage: '',
  backgroundFit: 'none',
});
// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); },flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
flowScheduler.add(preCue_InstructionRoutineBegin());
flowScheduler.add(preCue_InstructionRoutineEachFrame());
flowScheduler.add(preCue_InstructionRoutineEnd());
flowScheduler.add(Pre_Practice_windowRoutineBegin());
flowScheduler.add(Pre_Practice_windowRoutineEachFrame());
flowScheduler.add(Pre_Practice_windowRoutineEnd());
const practice_loopLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(practice_loopLoopBegin(practice_loopLoopScheduler));
flowScheduler.add(practice_loopLoopScheduler);
flowScheduler.add(practice_loopLoopEnd);



flowScheduler.add(Pre_experiment_windowRoutineBegin());
flowScheduler.add(Pre_experiment_windowRoutineEachFrame());
flowScheduler.add(Pre_experiment_windowRoutineEnd());
const trial_loopLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trial_loopLoopBegin(trial_loopLoopScheduler));
flowScheduler.add(trial_loopLoopScheduler);
flowScheduler.add(trial_loopLoopEnd);



flowScheduler.add(break_2RoutineBegin());
flowScheduler.add(break_2RoutineEachFrame());
flowScheduler.add(break_2RoutineEnd());
const trial_loop2LoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trial_loop2LoopBegin(trial_loop2LoopScheduler));
flowScheduler.add(trial_loop2LoopScheduler);
flowScheduler.add(trial_loop2LoopEnd);



flowScheduler.add(break_2RoutineBegin());
flowScheduler.add(break_2RoutineEachFrame());
flowScheduler.add(break_2RoutineEnd());
const trial_loop3LoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trial_loop3LoopBegin(trial_loop3LoopScheduler));
flowScheduler.add(trial_loop3LoopScheduler);
flowScheduler.add(trial_loop3LoopEnd);



flowScheduler.add(break_2RoutineBegin());
flowScheduler.add(break_2RoutineEachFrame());
flowScheduler.add(break_2RoutineEnd());
const trial_loop4LoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trial_loop4LoopBegin(trial_loop4LoopScheduler));
flowScheduler.add(trial_loop4LoopScheduler);
flowScheduler.add(trial_loop4LoopEnd);



flowScheduler.add(quitPsychoJS, '', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, '', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  resources: [
    // resources:
    {'name': 'practiceconditions.csv', 'path': 'practiceconditions.csv'},
    {'name': 'conditions.csv', 'path': 'conditions.csv'},
    {'name': 'conditions.csv', 'path': 'conditions.csv'},
    {'name': 'conditions.csv', 'path': 'conditions.csv'},
    {'name': 'conditions.csv', 'path': 'conditions.csv'},
  ]
});

psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.WARNING);


var currentLoop;
var frameDur;
async function updateInfo() {
  currentLoop = psychoJS.experiment;  // right now there are no loops
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2024.2.4';
  expInfo['OS'] = window.navigator.platform;


  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  

  
  psychoJS.experiment.dataFileName = (("." + "/") + `data/${expInfo["participant"]}_${expName}_${expInfo["date"]}`);
  psychoJS.experiment.field_separator = '\t';


  return Scheduler.Event.NEXT;
}


var preCue_InstructionClock;
var text_instr_2;
var nextButton;
var precue_bottom_txt;
var Pre_Practice_windowClock;
var text_norm_3;
var key_instruct_3;
var preExperimentTxt_2;
var practice_trialsClock;
var practice_fixation;
var practice_leftBox;
var practice_RightBox;
var practice_key_resp;
var practice_Cue;
var practice_target;
var practice_error_feedbackClock;
var msg;
var text_2;
var Pre_experiment_windowClock;
var text_norm;
var key_instruct;
var preExperimentTxt;
var trialsClock;
var fixation;
var leftBox;
var rightBox;
var key_resp;
var cue;
var target;
var error_feedbackClock;
var text;
var break_2Clock;
var text_norm_2;
var key_instruct_2;
var globalClock;
var routineTimer;
async function experimentInit() {
  // Initialize components for Routine "preCue_Instruction"
  preCue_InstructionClock = new util.Clock();
  text_instr_2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_instr_2',
    text: 'Focus your eyes on the plus sign at all times during the experiment.\n\nTwo circles will appear one after the other:\nThe first circle is the cue—please ignore it.\nThe second circle is the target—this is what you need to respond to.\n\nYour task:\nPress the left arrow key if the target appears in the left box.\nPress the right arrow key if the target appears in the right box.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0.25], draggable: false, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  nextButton = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  precue_bottom_txt = new visual.TextStim({
    win: psychoJS.window,
    name: 'precue_bottom_txt',
    text: 'Press Space to Continue',
    font: 'Arial',
    units: undefined, 
    pos: [0, (- 0.35)], draggable: false, height: 0.04,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -2.0 
  });
  
  // Initialize components for Routine "Pre_Practice_window"
  Pre_Practice_windowClock = new util.Clock();
  text_norm_3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_norm_3',
    text: 'In the next portion you will get 15 practice trials!\n\nRemember to:\nOnly respond to the second circle \nKeep your eyes on the plus sign.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.25], draggable: false, height: 0.1,  wrapWidth: 1.5, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  key_instruct_3 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  preExperimentTxt_2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'preExperimentTxt_2',
    text: 'Press either arrow to Start',
    font: 'Arial',
    units: undefined, 
    pos: [0, (- 0.35)], draggable: false, height: 0.04,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -2.0 
  });
  
  // Initialize components for Routine "practice_trials"
  practice_trialsClock = new util.Clock();
  practice_fixation = new visual.TextStim({
    win: psychoJS.window,
    name: 'practice_fixation',
    text: '+',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 0.0986,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  practice_leftBox = new visual.Rect ({
    win: psychoJS.window, name: 'practice_leftBox', units : 'height', 
    width: [2.128, 2.128][0], height: [2.128, 2.128][1],
    ori: 0.0, 
    pos: [(- 5.334), 0], 
    draggable: false, 
    anchor: 'center', 
    lineWidth: 3.0, 
    lineColor: new util.Color('white'), 
    fillColor: new util.Color([0.75, 0.75, 0.75]), 
    colorSpace: 'rgb', 
    opacity: 1.0, 
    depth: -2, 
    interpolate: true, 
  });
  
  practice_RightBox = new visual.Rect ({
    win: psychoJS.window, name: 'practice_RightBox', units : 'height', 
    width: [2.128, 2.128][0], height: [2.128, 2.128][1],
    ori: 0.0, 
    pos: [5.334, 0], 
    draggable: false, 
    anchor: 'center', 
    lineWidth: 3.0, 
    lineColor: new util.Color('white'), 
    fillColor: new util.Color('grey'), 
    colorSpace: 'rgb', 
    opacity: 1.0, 
    depth: -3, 
    interpolate: true, 
  });
  
  practice_key_resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  practice_Cue = new visual.Rect ({
    win: psychoJS.window, name: 'practice_Cue', units : 'height', 
    width: [2.128, 2.128][0], height: [2.128, 2.128][1],
    ori: 0.0, 
    pos: [0, 0], 
    draggable: false, 
    anchor: 'center', 
    lineWidth: 3.0, 
    lineColor: new util.Color('white'), 
    fillColor: new util.Color('white'), 
    colorSpace: 'rgb', 
    opacity: 1.0, 
    depth: -5, 
    interpolate: true, 
  });
  
  practice_target = new visual.Rect ({
    win: psychoJS.window, name: 'practice_target', units : 'height', 
    width: [2.128, 2.128][0], height: [2.128, 2.128][1],
    ori: 0.0, 
    pos: [0, 0], 
    draggable: false, 
    anchor: 'center', 
    lineWidth: 3.0, 
    lineColor: new util.Color('white'), 
    fillColor: new util.Color('white'), 
    colorSpace: 'rgb', 
    opacity: 1.0, 
    depth: -6, 
    interpolate: true, 
  });
  
  // Initialize components for Routine "practice_error_feedback"
  practice_error_feedbackClock = new util.Clock();
  // Run 'Begin Experiment' code from code_3
  msg = "";
  
  text_2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_2',
    text: '',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0.25], draggable: false, height: 0.075,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  // Initialize components for Routine "Pre_experiment_window"
  Pre_experiment_windowClock = new util.Clock();
  text_norm = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_norm',
    text: 'In the next portion you will begin the experiment\nEvery 128 trials you will get a break, after 4 total blocks the experiment will close out. \nRemember to:\nOnly respond to the second circle (target), and keep your eyes on the plus sign.',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0.25], draggable: false, height: 0.1,  wrapWidth: 1.5, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  key_instruct = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Run 'Begin Experiment' code from text_align
  // Code component set to Both
  text_norm.setAlignHoriz('left')
  preExperimentTxt = new visual.TextStim({
    win: psychoJS.window,
    name: 'preExperimentTxt',
    text: 'Press either arrow to Continue',
    font: 'Arial',
    units: undefined, 
    pos: [0, (- 0.35)], draggable: false, height: 0.04,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -3.0 
  });
  
  // Initialize components for Routine "trials"
  trialsClock = new util.Clock();
  fixation = new visual.TextStim({
    win: psychoJS.window,
    name: 'fixation',
    text: '+',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  leftBox = new visual.Rect ({
    win: psychoJS.window, name: 'leftBox', units : 'height', 
    width: [2.128, 2.128][0], height: [2.128, 2.128][1],
    ori: 0.0, 
    pos: [(- 5.334), 0], 
    draggable: false, 
    anchor: 'center', 
    lineWidth: 3.0, 
    lineColor: new util.Color('white'), 
    fillColor: new util.Color([(- 1.0), (- 1.0), (- 1.0)]), 
    colorSpace: 'rgb', 
    opacity: 0.5, 
    depth: -2, 
    interpolate: true, 
  });
  
  rightBox = new visual.Rect ({
    win: psychoJS.window, name: 'rightBox', units : 'height', 
    width: [2.128, 2.128][0], height: [2.128, 2.128][1],
    ori: 0.0, 
    pos: [5.334, 0], 
    draggable: false, 
    anchor: 'center', 
    lineWidth: 3.0, 
    lineColor: new util.Color('white'), 
    fillColor: new util.Color([(- 1.0), (- 1.0), (- 1.0)]), 
    colorSpace: 'rgb', 
    opacity: 0.5, 
    depth: -3, 
    interpolate: true, 
  });
  
  key_resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  cue = new visual.Rect ({
    win: psychoJS.window, name: 'cue', units : 'height', 
    width: [2.128, 2.128][0], height: [2.128, 2.128][1],
    ori: 0.0, 
    pos: [0, 0], 
    draggable: false, 
    anchor: 'center', 
    lineWidth: 3.0, 
    lineColor: new util.Color('white'), 
    fillColor: new util.Color('white'), 
    colorSpace: 'rgb', 
    opacity: 1.0, 
    depth: -5, 
    interpolate: true, 
  });
  
  target = new visual.Rect ({
    win: psychoJS.window, name: 'target', units : 'height', 
    width: [2.128, 2.128][0], height: [2.128, 2.128][1],
    ori: 0.0, 
    pos: [0, 0], 
    draggable: false, 
    anchor: 'center', 
    lineWidth: 3.0, 
    lineColor: new util.Color('white'), 
    fillColor: new util.Color('white'), 
    colorSpace: 'rgb', 
    opacity: 1.0, 
    depth: -6, 
    interpolate: true, 
  });
  
  // Initialize components for Routine "error_feedback"
  error_feedbackClock = new util.Clock();
  // Run 'Begin Experiment' code from code_2
  msg = "";
  
  text = new visual.TextStim({
    win: psychoJS.window,
    name: 'text',
    text: '',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0.25], draggable: false, height: 0.075,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  // Initialize components for Routine "break_2"
  break_2Clock = new util.Clock();
  text_norm_2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_norm_2',
    text: 'You can take a short break now.\n\n\n\nPress any arrow to continue',
    font: 'Arial',
    units: 'norm', 
    pos: [0, 0], draggable: false, height: 0.1,  wrapWidth: 1.8, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  key_instruct_2 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}


var t;
var frameN;
var continueRoutine;
var preCue_InstructionMaxDurationReached;
var _nextButton_allKeys;
var preCue_InstructionMaxDuration;
var preCue_InstructionStartWinParams;
var preCue_InstructionComponents;
function preCue_InstructionRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'preCue_Instruction' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    preCue_InstructionClock.reset();
    routineTimer.reset();
    preCue_InstructionMaxDurationReached = false;
    // update component parameters for each repeat
    nextButton.keys = undefined;
    nextButton.rt = undefined;
    _nextButton_allKeys = [];
    psychoJS.experiment.addData('preCue_Instruction.started', globalClock.getTime());
    preCue_InstructionMaxDuration = null
    preCue_InstructionStartWinParams = {
        'color': psychoJS.window.color,
        'colorSpace': psychoJS.window.colorSpace,
        'backgroundImage': psychoJS.window.backgroundImage,
        'backgroundFit': psychoJS.window.backgroundFit,
    };
    psychoJS.window.color = [(- 1.0), (- 1.0), (- 1.0)];
    psychoJS.window.colorSpace = 'rgb';
    psychoJS.window.backgroundImage = '';
    psychoJS.window.backgroundFit = 'none';
    // keep track of which components have finished
    preCue_InstructionComponents = [];
    preCue_InstructionComponents.push(text_instr_2);
    preCue_InstructionComponents.push(nextButton);
    preCue_InstructionComponents.push(precue_bottom_txt);
    
    for (const thisComponent of preCue_InstructionComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function preCue_InstructionRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'preCue_Instruction' ---
    // get current time
    t = preCue_InstructionClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_instr_2* updates
    if (t >= 0.0 && text_instr_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_instr_2.tStart = t;  // (not accounting for frame time here)
      text_instr_2.frameNStart = frameN;  // exact frame index
      
      text_instr_2.setAutoDraw(true);
    }
    
    
    // *nextButton* updates
    if (t >= 0.0 && nextButton.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      nextButton.tStart = t;  // (not accounting for frame time here)
      nextButton.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { nextButton.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { nextButton.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { nextButton.clearEvents(); });
    }
    
    if (nextButton.status === PsychoJS.Status.STARTED) {
      let theseKeys = nextButton.getKeys({keyList: ['space'], waitRelease: false});
      _nextButton_allKeys = _nextButton_allKeys.concat(theseKeys);
      if (_nextButton_allKeys.length > 0) {
        nextButton.keys = _nextButton_allKeys[0].name;  // just the first key pressed
        nextButton.rt = _nextButton_allKeys[0].rt;
        nextButton.duration = _nextButton_allKeys[0].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *precue_bottom_txt* updates
    if (t >= 0.0 && precue_bottom_txt.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      precue_bottom_txt.tStart = t;  // (not accounting for frame time here)
      precue_bottom_txt.frameNStart = frameN;  // exact frame index
      
      precue_bottom_txt.setAutoDraw(true);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of preCue_InstructionComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function preCue_InstructionRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'preCue_Instruction' ---
    for (const thisComponent of preCue_InstructionComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('preCue_Instruction.stopped', globalClock.getTime());
    psychoJS.window.color = preCue_InstructionStartWinParams['color'];
    psychoJS.window.colorSpace = preCue_InstructionStartWinParams['colorSpace'];
    psychoJS.window.backgroundImage = preCue_InstructionStartWinParams['backgroundImage'];
    psychoJS.window.backgroundFit = preCue_InstructionStartWinParams['backgroundFit'];
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(nextButton.corr, level);
    }
    psychoJS.experiment.addData('nextButton.keys', nextButton.keys);
    if (typeof nextButton.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('nextButton.rt', nextButton.rt);
        psychoJS.experiment.addData('nextButton.duration', nextButton.duration);
        routineTimer.reset();
        }
    
    nextButton.stop();
    // the Routine "preCue_Instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var Pre_Practice_windowMaxDurationReached;
var _key_instruct_3_allKeys;
var Pre_Practice_windowMaxDuration;
var Pre_Practice_windowStartWinParams;
var Pre_Practice_windowComponents;
function Pre_Practice_windowRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Pre_Practice_window' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    Pre_Practice_windowClock.reset();
    routineTimer.reset();
    Pre_Practice_windowMaxDurationReached = false;
    // update component parameters for each repeat
    key_instruct_3.keys = undefined;
    key_instruct_3.rt = undefined;
    _key_instruct_3_allKeys = [];
    psychoJS.experiment.addData('Pre_Practice_window.started', globalClock.getTime());
    Pre_Practice_windowMaxDuration = null
    Pre_Practice_windowStartWinParams = {
        'color': psychoJS.window.color,
        'colorSpace': psychoJS.window.colorSpace,
        'backgroundImage': psychoJS.window.backgroundImage,
        'backgroundFit': psychoJS.window.backgroundFit,
    };
    psychoJS.window.color = [(- 1.0), (- 1.0), (- 1.0)];
    psychoJS.window.colorSpace = 'rgb';
    psychoJS.window.backgroundImage = '';
    psychoJS.window.backgroundFit = 'none';
    // keep track of which components have finished
    Pre_Practice_windowComponents = [];
    Pre_Practice_windowComponents.push(text_norm_3);
    Pre_Practice_windowComponents.push(key_instruct_3);
    Pre_Practice_windowComponents.push(preExperimentTxt_2);
    
    for (const thisComponent of Pre_Practice_windowComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function Pre_Practice_windowRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Pre_Practice_window' ---
    // get current time
    t = Pre_Practice_windowClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_norm_3* updates
    if (t >= 0.0 && text_norm_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_norm_3.tStart = t;  // (not accounting for frame time here)
      text_norm_3.frameNStart = frameN;  // exact frame index
      
      text_norm_3.setAutoDraw(true);
    }
    
    
    // *key_instruct_3* updates
    if (t >= 0.0 && key_instruct_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_instruct_3.tStart = t;  // (not accounting for frame time here)
      key_instruct_3.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_instruct_3.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_instruct_3.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_instruct_3.clearEvents(); });
    }
    
    if (key_instruct_3.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_instruct_3.getKeys({keyList: ['left', 'right'], waitRelease: false});
      _key_instruct_3_allKeys = _key_instruct_3_allKeys.concat(theseKeys);
      if (_key_instruct_3_allKeys.length > 0) {
        key_instruct_3.keys = _key_instruct_3_allKeys[0].name;  // just the first key pressed
        key_instruct_3.rt = _key_instruct_3_allKeys[0].rt;
        key_instruct_3.duration = _key_instruct_3_allKeys[0].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *preExperimentTxt_2* updates
    if (t >= 0.0 && preExperimentTxt_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      preExperimentTxt_2.tStart = t;  // (not accounting for frame time here)
      preExperimentTxt_2.frameNStart = frameN;  // exact frame index
      
      preExperimentTxt_2.setAutoDraw(true);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of Pre_Practice_windowComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Pre_Practice_windowRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Pre_Practice_window' ---
    for (const thisComponent of Pre_Practice_windowComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('Pre_Practice_window.stopped', globalClock.getTime());
    psychoJS.window.color = Pre_Practice_windowStartWinParams['color'];
    psychoJS.window.colorSpace = Pre_Practice_windowStartWinParams['colorSpace'];
    psychoJS.window.backgroundImage = Pre_Practice_windowStartWinParams['backgroundImage'];
    psychoJS.window.backgroundFit = Pre_Practice_windowStartWinParams['backgroundFit'];
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_instruct_3.corr, level);
    }
    psychoJS.experiment.addData('key_instruct_3.keys', key_instruct_3.keys);
    if (typeof key_instruct_3.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_instruct_3.rt', key_instruct_3.rt);
        psychoJS.experiment.addData('key_instruct_3.duration', key_instruct_3.duration);
        routineTimer.reset();
        }
    
    key_instruct_3.stop();
    // the Routine "Pre_Practice_window" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var practice_loop;
function practice_loopLoopBegin(practice_loopLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    practice_loop = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 1, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'practiceconditions.csv',
      seed: undefined, name: 'practice_loop'
    });
    psychoJS.experiment.addLoop(practice_loop); // add the loop to the experiment
    currentLoop = practice_loop;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisPractice_loop of practice_loop) {
      snapshot = practice_loop.getSnapshot();
      practice_loopLoopScheduler.add(importConditions(snapshot));
      practice_loopLoopScheduler.add(practice_trialsRoutineBegin(snapshot));
      practice_loopLoopScheduler.add(practice_trialsRoutineEachFrame());
      practice_loopLoopScheduler.add(practice_trialsRoutineEnd(snapshot));
      practice_loopLoopScheduler.add(practice_error_feedbackRoutineBegin(snapshot));
      practice_loopLoopScheduler.add(practice_error_feedbackRoutineEachFrame());
      practice_loopLoopScheduler.add(practice_error_feedbackRoutineEnd(snapshot));
      practice_loopLoopScheduler.add(practice_loopLoopEndIteration(practice_loopLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}


async function practice_loopLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(practice_loop);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}


function practice_loopLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}


var trial_loop;
function trial_loopLoopBegin(trial_loopLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    trial_loop = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 2, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'conditions.csv',
      seed: undefined, name: 'trial_loop'
    });
    psychoJS.experiment.addLoop(trial_loop); // add the loop to the experiment
    currentLoop = trial_loop;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisTrial_loop of trial_loop) {
      snapshot = trial_loop.getSnapshot();
      trial_loopLoopScheduler.add(importConditions(snapshot));
      trial_loopLoopScheduler.add(trialsRoutineBegin(snapshot));
      trial_loopLoopScheduler.add(trialsRoutineEachFrame());
      trial_loopLoopScheduler.add(trialsRoutineEnd(snapshot));
      trial_loopLoopScheduler.add(error_feedbackRoutineBegin(snapshot));
      trial_loopLoopScheduler.add(error_feedbackRoutineEachFrame());
      trial_loopLoopScheduler.add(error_feedbackRoutineEnd(snapshot));
      trial_loopLoopScheduler.add(trial_loopLoopEndIteration(trial_loopLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}


async function trial_loopLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(trial_loop);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}


function trial_loopLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}


var trial_loop2;
function trial_loop2LoopBegin(trial_loop2LoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    trial_loop2 = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 2, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'conditions.csv',
      seed: undefined, name: 'trial_loop2'
    });
    psychoJS.experiment.addLoop(trial_loop2); // add the loop to the experiment
    currentLoop = trial_loop2;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisTrial_loop2 of trial_loop2) {
      snapshot = trial_loop2.getSnapshot();
      trial_loop2LoopScheduler.add(importConditions(snapshot));
      trial_loop2LoopScheduler.add(trialsRoutineBegin(snapshot));
      trial_loop2LoopScheduler.add(trialsRoutineEachFrame());
      trial_loop2LoopScheduler.add(trialsRoutineEnd(snapshot));
      trial_loop2LoopScheduler.add(error_feedbackRoutineBegin(snapshot));
      trial_loop2LoopScheduler.add(error_feedbackRoutineEachFrame());
      trial_loop2LoopScheduler.add(error_feedbackRoutineEnd(snapshot));
      trial_loop2LoopScheduler.add(trial_loop2LoopEndIteration(trial_loop2LoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}


async function trial_loop2LoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(trial_loop2);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}


function trial_loop2LoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}


var trial_loop3;
function trial_loop3LoopBegin(trial_loop3LoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    trial_loop3 = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 2, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'conditions.csv',
      seed: undefined, name: 'trial_loop3'
    });
    psychoJS.experiment.addLoop(trial_loop3); // add the loop to the experiment
    currentLoop = trial_loop3;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisTrial_loop3 of trial_loop3) {
      snapshot = trial_loop3.getSnapshot();
      trial_loop3LoopScheduler.add(importConditions(snapshot));
      trial_loop3LoopScheduler.add(trialsRoutineBegin(snapshot));
      trial_loop3LoopScheduler.add(trialsRoutineEachFrame());
      trial_loop3LoopScheduler.add(trialsRoutineEnd(snapshot));
      trial_loop3LoopScheduler.add(error_feedbackRoutineBegin(snapshot));
      trial_loop3LoopScheduler.add(error_feedbackRoutineEachFrame());
      trial_loop3LoopScheduler.add(error_feedbackRoutineEnd(snapshot));
      trial_loop3LoopScheduler.add(trial_loop3LoopEndIteration(trial_loop3LoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}


async function trial_loop3LoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(trial_loop3);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}


function trial_loop3LoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}


var trial_loop4;
function trial_loop4LoopBegin(trial_loop4LoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    trial_loop4 = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 2, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'conditions.csv',
      seed: undefined, name: 'trial_loop4'
    });
    psychoJS.experiment.addLoop(trial_loop4); // add the loop to the experiment
    currentLoop = trial_loop4;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisTrial_loop4 of trial_loop4) {
      snapshot = trial_loop4.getSnapshot();
      trial_loop4LoopScheduler.add(importConditions(snapshot));
      trial_loop4LoopScheduler.add(trialsRoutineBegin(snapshot));
      trial_loop4LoopScheduler.add(trialsRoutineEachFrame());
      trial_loop4LoopScheduler.add(trialsRoutineEnd(snapshot));
      trial_loop4LoopScheduler.add(error_feedbackRoutineBegin(snapshot));
      trial_loop4LoopScheduler.add(error_feedbackRoutineEachFrame());
      trial_loop4LoopScheduler.add(error_feedbackRoutineEnd(snapshot));
      trial_loop4LoopScheduler.add(trial_loop4LoopEndIteration(trial_loop4LoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}


async function trial_loop4LoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(trial_loop4);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}


function trial_loop4LoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}


var practice_trialsMaxDurationReached;
var _practice_key_resp_allKeys;
var practice_trialsMaxDuration;
var practice_trialsStartWinParams;
var practice_trialsComponents;
function practice_trialsRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'practice_trials' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    practice_trialsClock.reset();
    routineTimer.reset();
    practice_trialsMaxDurationReached = false;
    // update component parameters for each repeat
    practice_key_resp.keys = undefined;
    practice_key_resp.rt = undefined;
    _practice_key_resp_allKeys = [];
    practice_Cue.setPos(cueSide);
    practice_target.setPos(targetSide);
    psychoJS.experiment.addData('practice_trials.started', globalClock.getTime());
    practice_trialsMaxDuration = null
    practice_trialsStartWinParams = {
        'color': psychoJS.window.color,
        'colorSpace': psychoJS.window.colorSpace,
        'backgroundImage': psychoJS.window.backgroundImage,
        'backgroundFit': psychoJS.window.backgroundFit,
    };
    psychoJS.window.color = [(- 1.0), (- 1.0), (- 1.0)];
    psychoJS.window.colorSpace = 'rgb';
    psychoJS.window.backgroundImage = '';
    psychoJS.window.backgroundFit = 'fill';
    // keep track of which components have finished
    practice_trialsComponents = [];
    practice_trialsComponents.push(practice_fixation);
    practice_trialsComponents.push(practice_leftBox);
    practice_trialsComponents.push(practice_RightBox);
    practice_trialsComponents.push(practice_key_resp);
    practice_trialsComponents.push(practice_Cue);
    practice_trialsComponents.push(practice_target);
    
    for (const thisComponent of practice_trialsComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


var frameRemains;
function practice_trialsRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'practice_trials' ---
    // get current time
    t = practice_trialsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *practice_fixation* updates
    if (t >= 0.0 && practice_fixation.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      practice_fixation.tStart = t;  // (not accounting for frame time here)
      practice_fixation.frameNStart = frameN;  // exact frame index
      
      practice_fixation.setAutoDraw(true);
    }
    
    frameRemains = ((0.8 + ISI) + 1) - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if ((practice_fixation.status === PsychoJS.Status.STARTED || practice_fixation.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      practice_fixation.setAutoDraw(false);
    }
    
    
    if (practice_leftBox.status === PsychoJS.Status.STARTED){ // only update if being drawn
      practice_leftBox.setLineColor(new util.Color('grey'), false);
    }
    
    // *practice_leftBox* updates
    if (t >= 0.0 && practice_leftBox.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      practice_leftBox.tStart = t;  // (not accounting for frame time here)
      practice_leftBox.frameNStart = frameN;  // exact frame index
      
      practice_leftBox.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + ((0.8 + ISI) + 1) - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (practice_leftBox.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      practice_leftBox.setAutoDraw(false);
    }
    
    
    if (practice_RightBox.status === PsychoJS.Status.STARTED){ // only update if being drawn
      practice_RightBox.setLineColor(new util.Color('grey'), false);
    }
    
    // *practice_RightBox* updates
    if (t >= 0.0 && practice_RightBox.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      practice_RightBox.tStart = t;  // (not accounting for frame time here)
      practice_RightBox.frameNStart = frameN;  // exact frame index
      
      practice_RightBox.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + ((0.8 + ISI) + 1) - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (practice_RightBox.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      practice_RightBox.setAutoDraw(false);
    }
    
    
    // *practice_key_resp* updates
    if (t >= 0.0 && practice_key_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      practice_key_resp.tStart = t;  // (not accounting for frame time here)
      practice_key_resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { practice_key_resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { practice_key_resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { practice_key_resp.clearEvents(); });
    }
    
    frameRemains = 0.0 + ((0.8 + ISI) + 1) - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (practice_key_resp.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      practice_key_resp.status = PsychoJS.Status.FINISHED;
        }
      
    if (practice_key_resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = practice_key_resp.getKeys({keyList: ['left', 'right'], waitRelease: false});
      _practice_key_resp_allKeys = _practice_key_resp_allKeys.concat(theseKeys);
      if (_practice_key_resp_allKeys.length > 0) {
        practice_key_resp.keys = _practice_key_resp_allKeys[0].name;  // just the first key pressed
        practice_key_resp.rt = _practice_key_resp_allKeys[0].rt;
        practice_key_resp.duration = _practice_key_resp_allKeys[0].duration;
        // was this correct?
        if (practice_key_resp.keys == correct) {
            practice_key_resp.corr = 1;
        } else {
            practice_key_resp.corr = 0;
        }
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *practice_Cue* updates
    if (t >= 0.75 && practice_Cue.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      practice_Cue.tStart = t;  // (not accounting for frame time here)
      practice_Cue.frameNStart = frameN;  // exact frame index
      
      practice_Cue.setAutoDraw(true);
    }
    
    frameRemains = 0.75 + 0.05 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (practice_Cue.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      practice_Cue.setAutoDraw(false);
    }
    
    
    // *practice_target* updates
    if (t >= (0.8 + ISI) && practice_target.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      practice_target.tStart = t;  // (not accounting for frame time here)
      practice_target.frameNStart = frameN;  // exact frame index
      
      practice_target.setAutoDraw(true);
    }
    
    frameRemains = (0.8 + ISI) + 1 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (practice_target.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      practice_target.setAutoDraw(false);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of practice_trialsComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function practice_trialsRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'practice_trials' ---
    for (const thisComponent of practice_trialsComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('practice_trials.stopped', globalClock.getTime());
    psychoJS.window.color = practice_trialsStartWinParams['color'];
    psychoJS.window.colorSpace = practice_trialsStartWinParams['colorSpace'];
    psychoJS.window.backgroundImage = practice_trialsStartWinParams['backgroundImage'];
    psychoJS.window.backgroundFit = practice_trialsStartWinParams['backgroundFit'];
    // was no response the correct answer?!
    if (practice_key_resp.keys === undefined) {
      if (['None','none',undefined].includes(correct)) {
         practice_key_resp.corr = 1;  // correct non-response
      } else {
         practice_key_resp.corr = 0;  // failed to respond (incorrectly)
      }
    }
    // store data for current loop
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(practice_key_resp.corr, level);
    }
    psychoJS.experiment.addData('practice_key_resp.keys', practice_key_resp.keys);
    psychoJS.experiment.addData('practice_key_resp.corr', practice_key_resp.corr);
    if (typeof practice_key_resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('practice_key_resp.rt', practice_key_resp.rt);
        psychoJS.experiment.addData('practice_key_resp.duration', practice_key_resp.duration);
        routineTimer.reset();
        }
    
    practice_key_resp.stop();
    // the Routine "practice_trials" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var practice_error_feedbackMaxDurationReached;
var practice_error_feedbackMaxDuration;
var practice_error_feedbackStartWinParams;
var practice_error_feedbackComponents;
function practice_error_feedbackRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'practice_error_feedback' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    practice_error_feedbackClock.reset(routineTimer.getTime());
    routineTimer.add(1.000000);
    practice_error_feedbackMaxDurationReached = false;
    // update component parameters for each repeat
    // Run 'Begin Routine' code from code_3
    if ((! practice_key_resp.keys)) {
        msg = "Too slow! Please respond faster.";
    } else {
        msg = "";
    }
    
    text_2.setColor(new util.Color('red'));
    text_2.setText(msg);
    psychoJS.experiment.addData('practice_error_feedback.started', globalClock.getTime());
    practice_error_feedbackMaxDuration = null
    practice_error_feedbackStartWinParams = {
        'color': psychoJS.window.color,
        'colorSpace': psychoJS.window.colorSpace,
        'backgroundImage': psychoJS.window.backgroundImage,
        'backgroundFit': psychoJS.window.backgroundFit,
    };
    psychoJS.window.color = [(- 1.0), (- 1.0), (- 1.0)];
    psychoJS.window.colorSpace = 'rgb';
    psychoJS.window.backgroundImage = '';
    psychoJS.window.backgroundFit = 'none';
    // keep track of which components have finished
    practice_error_feedbackComponents = [];
    practice_error_feedbackComponents.push(text_2);
    
    for (const thisComponent of practice_error_feedbackComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function practice_error_feedbackRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'practice_error_feedback' ---
    // get current time
    t = practice_error_feedbackClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_2* updates
    if (t >= 0 && text_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_2.tStart = t;  // (not accounting for frame time here)
      text_2.frameNStart = frameN;  // exact frame index
      
      text_2.setAutoDraw(true);
    }
    
    frameRemains = 0 + 1 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (text_2.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      text_2.setAutoDraw(false);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of practice_error_feedbackComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function practice_error_feedbackRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'practice_error_feedback' ---
    for (const thisComponent of practice_error_feedbackComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('practice_error_feedback.stopped', globalClock.getTime());
    psychoJS.window.color = practice_error_feedbackStartWinParams['color'];
    psychoJS.window.colorSpace = practice_error_feedbackStartWinParams['colorSpace'];
    psychoJS.window.backgroundImage = practice_error_feedbackStartWinParams['backgroundImage'];
    psychoJS.window.backgroundFit = practice_error_feedbackStartWinParams['backgroundFit'];
    if (practice_error_feedbackMaxDurationReached) {
        practice_error_feedbackClock.add(practice_error_feedbackMaxDuration);
    } else {
        practice_error_feedbackClock.add(1.000000);
    }
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var Pre_experiment_windowMaxDurationReached;
var _key_instruct_allKeys;
var Pre_experiment_windowMaxDuration;
var Pre_experiment_windowStartWinParams;
var Pre_experiment_windowComponents;
function Pre_experiment_windowRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Pre_experiment_window' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    Pre_experiment_windowClock.reset();
    routineTimer.reset();
    Pre_experiment_windowMaxDurationReached = false;
    // update component parameters for each repeat
    key_instruct.keys = undefined;
    key_instruct.rt = undefined;
    _key_instruct_allKeys = [];
    psychoJS.experiment.addData('Pre_experiment_window.started', globalClock.getTime());
    Pre_experiment_windowMaxDuration = null
    Pre_experiment_windowStartWinParams = {
        'color': psychoJS.window.color,
        'colorSpace': psychoJS.window.colorSpace,
        'backgroundImage': psychoJS.window.backgroundImage,
        'backgroundFit': psychoJS.window.backgroundFit,
    };
    psychoJS.window.color = [(- 1.0), (- 1.0), (- 1.0)];
    psychoJS.window.colorSpace = 'rgb';
    psychoJS.window.backgroundImage = '';
    psychoJS.window.backgroundFit = 'none';
    // keep track of which components have finished
    Pre_experiment_windowComponents = [];
    Pre_experiment_windowComponents.push(text_norm);
    Pre_experiment_windowComponents.push(key_instruct);
    Pre_experiment_windowComponents.push(preExperimentTxt);
    
    for (const thisComponent of Pre_experiment_windowComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function Pre_experiment_windowRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Pre_experiment_window' ---
    // get current time
    t = Pre_experiment_windowClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_norm* updates
    if (t >= 0.0 && text_norm.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_norm.tStart = t;  // (not accounting for frame time here)
      text_norm.frameNStart = frameN;  // exact frame index
      
      text_norm.setAutoDraw(true);
    }
    
    
    // *key_instruct* updates
    if (t >= 0.0 && key_instruct.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_instruct.tStart = t;  // (not accounting for frame time here)
      key_instruct.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_instruct.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_instruct.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_instruct.clearEvents(); });
    }
    
    if (key_instruct.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_instruct.getKeys({keyList: ['left', 'right'], waitRelease: false});
      _key_instruct_allKeys = _key_instruct_allKeys.concat(theseKeys);
      if (_key_instruct_allKeys.length > 0) {
        key_instruct.keys = _key_instruct_allKeys[0].name;  // just the first key pressed
        key_instruct.rt = _key_instruct_allKeys[0].rt;
        key_instruct.duration = _key_instruct_allKeys[0].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *preExperimentTxt* updates
    if (t >= 0.0 && preExperimentTxt.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      preExperimentTxt.tStart = t;  // (not accounting for frame time here)
      preExperimentTxt.frameNStart = frameN;  // exact frame index
      
      preExperimentTxt.setAutoDraw(true);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of Pre_experiment_windowComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Pre_experiment_windowRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Pre_experiment_window' ---
    for (const thisComponent of Pre_experiment_windowComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('Pre_experiment_window.stopped', globalClock.getTime());
    psychoJS.window.color = Pre_experiment_windowStartWinParams['color'];
    psychoJS.window.colorSpace = Pre_experiment_windowStartWinParams['colorSpace'];
    psychoJS.window.backgroundImage = Pre_experiment_windowStartWinParams['backgroundImage'];
    psychoJS.window.backgroundFit = Pre_experiment_windowStartWinParams['backgroundFit'];
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_instruct.corr, level);
    }
    psychoJS.experiment.addData('key_instruct.keys', key_instruct.keys);
    if (typeof key_instruct.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_instruct.rt', key_instruct.rt);
        psychoJS.experiment.addData('key_instruct.duration', key_instruct.duration);
        routineTimer.reset();
        }
    
    key_instruct.stop();
    // the Routine "Pre_experiment_window" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var trialsMaxDurationReached;
var _key_resp_allKeys;
var trialsMaxDuration;
var trialsStartWinParams;
var trialsComponents;
function trialsRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'trials' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    trialsClock.reset();
    routineTimer.reset();
    trialsMaxDurationReached = false;
    // update component parameters for each repeat
    key_resp.keys = undefined;
    key_resp.rt = undefined;
    _key_resp_allKeys = [];
    cue.setPos(cueSide);
    target.setPos(targetSide);
    psychoJS.experiment.addData('trials.started', globalClock.getTime());
    trialsMaxDuration = null
    trialsStartWinParams = {
        'color': psychoJS.window.color,
        'colorSpace': psychoJS.window.colorSpace,
        'backgroundImage': psychoJS.window.backgroundImage,
        'backgroundFit': psychoJS.window.backgroundFit,
    };
    psychoJS.window.color = [(- 1.0), (- 1.0), (- 1.0)];
    psychoJS.window.colorSpace = 'rgb';
    psychoJS.window.backgroundImage = '';
    psychoJS.window.backgroundFit = 'none';
    // keep track of which components have finished
    trialsComponents = [];
    trialsComponents.push(fixation);
    trialsComponents.push(leftBox);
    trialsComponents.push(rightBox);
    trialsComponents.push(key_resp);
    trialsComponents.push(cue);
    trialsComponents.push(target);
    
    for (const thisComponent of trialsComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function trialsRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'trials' ---
    // get current time
    t = trialsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *fixation* updates
    if (t >= 0.0 && fixation.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      fixation.tStart = t;  // (not accounting for frame time here)
      fixation.frameNStart = frameN;  // exact frame index
      
      fixation.setAutoDraw(true);
    }
    
    frameRemains = ((0.8 + ISI) + 1) - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if ((fixation.status === PsychoJS.Status.STARTED || fixation.status === PsychoJS.Status.FINISHED) && t >= frameRemains) {
      fixation.setAutoDraw(false);
    }
    
    
    if (leftBox.status === PsychoJS.Status.STARTED){ // only update if being drawn
      leftBox.setLineColor(new util.Color([(- 1.0), (- 1.0), (- 1.0)]), false);
    }
    
    // *leftBox* updates
    if (t >= 0.0 && leftBox.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      leftBox.tStart = t;  // (not accounting for frame time here)
      leftBox.frameNStart = frameN;  // exact frame index
      
      leftBox.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + ((0.8 + ISI) + 1) - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (leftBox.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      leftBox.setAutoDraw(false);
    }
    
    
    if (rightBox.status === PsychoJS.Status.STARTED){ // only update if being drawn
      rightBox.setLineColor(new util.Color([(- 1.0), (- 1.0), (- 1.0)]), false);
    }
    
    // *rightBox* updates
    if (t >= 0.0 && rightBox.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      rightBox.tStart = t;  // (not accounting for frame time here)
      rightBox.frameNStart = frameN;  // exact frame index
      
      rightBox.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + ((0.8 + ISI) + 1) - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (rightBox.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      rightBox.setAutoDraw(false);
    }
    
    
    // *key_resp* updates
    if (t >= 0.0 && key_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp.tStart = t;  // (not accounting for frame time here)
      key_resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp.clearEvents(); });
    }
    
    frameRemains = 0.0 + ((0.8 + ISI) + 1) - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (key_resp.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      key_resp.status = PsychoJS.Status.FINISHED;
        }
      
    if (key_resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp.getKeys({keyList: ['left', 'right'], waitRelease: false});
      _key_resp_allKeys = _key_resp_allKeys.concat(theseKeys);
      if (_key_resp_allKeys.length > 0) {
        key_resp.keys = _key_resp_allKeys[0].name;  // just the first key pressed
        key_resp.rt = _key_resp_allKeys[0].rt;
        key_resp.duration = _key_resp_allKeys[0].duration;
        // was this correct?
        if (key_resp.keys == correct) {
            key_resp.corr = 1;
        } else {
            key_resp.corr = 0;
        }
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *cue* updates
    if (t >= 0.75 && cue.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      cue.tStart = t;  // (not accounting for frame time here)
      cue.frameNStart = frameN;  // exact frame index
      
      cue.setAutoDraw(true);
    }
    
    frameRemains = 0.75 + 0.05 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (cue.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      cue.setAutoDraw(false);
    }
    
    
    // *target* updates
    if (t >= (0.8 + ISI) && target.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      target.tStart = t;  // (not accounting for frame time here)
      target.frameNStart = frameN;  // exact frame index
      
      target.setAutoDraw(true);
    }
    
    frameRemains = (0.8 + ISI) + 1 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (target.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      target.setAutoDraw(false);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of trialsComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function trialsRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'trials' ---
    for (const thisComponent of trialsComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('trials.stopped', globalClock.getTime());
    psychoJS.window.color = trialsStartWinParams['color'];
    psychoJS.window.colorSpace = trialsStartWinParams['colorSpace'];
    psychoJS.window.backgroundImage = trialsStartWinParams['backgroundImage'];
    psychoJS.window.backgroundFit = trialsStartWinParams['backgroundFit'];
    // was no response the correct answer?!
    if (key_resp.keys === undefined) {
      if (['None','none',undefined].includes(correct)) {
         key_resp.corr = 1;  // correct non-response
      } else {
         key_resp.corr = 0;  // failed to respond (incorrectly)
      }
    }
    // store data for current loop
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp.corr, level);
    }
    psychoJS.experiment.addData('key_resp.keys', key_resp.keys);
    psychoJS.experiment.addData('key_resp.corr', key_resp.corr);
    if (typeof key_resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp.rt', key_resp.rt);
        psychoJS.experiment.addData('key_resp.duration', key_resp.duration);
        routineTimer.reset();
        }
    
    key_resp.stop();
    // the Routine "trials" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var error_feedbackMaxDurationReached;
var error_feedbackMaxDuration;
var error_feedbackStartWinParams;
var error_feedbackComponents;
function error_feedbackRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'error_feedback' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    error_feedbackClock.reset(routineTimer.getTime());
    routineTimer.add(1.000000);
    error_feedbackMaxDurationReached = false;
    // update component parameters for each repeat
    // Run 'Begin Routine' code from code_2
    if ((! key_resp.keys)) {
        msg = "Too slow! Please respond faster.";
    } else {
        msg = "";
    }
    
    text.setColor(new util.Color([1.0, (- 1.0), (- 1.0)]));
    text.setText(msg);
    psychoJS.experiment.addData('error_feedback.started', globalClock.getTime());
    error_feedbackMaxDuration = null
    error_feedbackStartWinParams = {
        'color': psychoJS.window.color,
        'colorSpace': psychoJS.window.colorSpace,
        'backgroundImage': psychoJS.window.backgroundImage,
        'backgroundFit': psychoJS.window.backgroundFit,
    };
    psychoJS.window.color = [(- 1.0), (- 1.0), (- 1.0)];
    psychoJS.window.colorSpace = 'rgb';
    psychoJS.window.backgroundImage = '';
    psychoJS.window.backgroundFit = 'none';
    // keep track of which components have finished
    error_feedbackComponents = [];
    error_feedbackComponents.push(text);
    
    for (const thisComponent of error_feedbackComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function error_feedbackRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'error_feedback' ---
    // get current time
    t = error_feedbackClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text* updates
    if (t >= 0 && text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text.tStart = t;  // (not accounting for frame time here)
      text.frameNStart = frameN;  // exact frame index
      
      text.setAutoDraw(true);
    }
    
    frameRemains = 0 + 1 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (text.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      text.setAutoDraw(false);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of error_feedbackComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function error_feedbackRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'error_feedback' ---
    for (const thisComponent of error_feedbackComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('error_feedback.stopped', globalClock.getTime());
    psychoJS.window.color = error_feedbackStartWinParams['color'];
    psychoJS.window.colorSpace = error_feedbackStartWinParams['colorSpace'];
    psychoJS.window.backgroundImage = error_feedbackStartWinParams['backgroundImage'];
    psychoJS.window.backgroundFit = error_feedbackStartWinParams['backgroundFit'];
    if (error_feedbackMaxDurationReached) {
        error_feedbackClock.add(error_feedbackMaxDuration);
    } else {
        error_feedbackClock.add(1.000000);
    }
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var break_2MaxDurationReached;
var _key_instruct_2_allKeys;
var break_2MaxDuration;
var break_2Components;
function break_2RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'break_2' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    break_2Clock.reset();
    routineTimer.reset();
    break_2MaxDurationReached = false;
    // update component parameters for each repeat
    key_instruct_2.keys = undefined;
    key_instruct_2.rt = undefined;
    _key_instruct_2_allKeys = [];
    psychoJS.experiment.addData('break_2.started', globalClock.getTime());
    break_2MaxDuration = null
    // keep track of which components have finished
    break_2Components = [];
    break_2Components.push(text_norm_2);
    break_2Components.push(key_instruct_2);
    
    for (const thisComponent of break_2Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function break_2RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'break_2' ---
    // get current time
    t = break_2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_norm_2* updates
    if (t >= 0.0 && text_norm_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_norm_2.tStart = t;  // (not accounting for frame time here)
      text_norm_2.frameNStart = frameN;  // exact frame index
      
      text_norm_2.setAutoDraw(true);
    }
    
    
    // *key_instruct_2* updates
    if (t >= 0.0 && key_instruct_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_instruct_2.tStart = t;  // (not accounting for frame time here)
      key_instruct_2.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_instruct_2.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_instruct_2.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_instruct_2.clearEvents(); });
    }
    
    if (key_instruct_2.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_instruct_2.getKeys({keyList: ['left', 'right'], waitRelease: false});
      _key_instruct_2_allKeys = _key_instruct_2_allKeys.concat(theseKeys);
      if (_key_instruct_2_allKeys.length > 0) {
        key_instruct_2.keys = _key_instruct_2_allKeys[0].name;  // just the first key pressed
        key_instruct_2.rt = _key_instruct_2_allKeys[0].rt;
        key_instruct_2.duration = _key_instruct_2_allKeys[0].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of break_2Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function break_2RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'break_2' ---
    for (const thisComponent of break_2Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('break_2.stopped', globalClock.getTime());
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_instruct_2.corr, level);
    }
    psychoJS.experiment.addData('key_instruct_2.keys', key_instruct_2.keys);
    if (typeof key_instruct_2.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_instruct_2.rt', key_instruct_2.rt);
        psychoJS.experiment.addData('key_instruct_2.duration', key_instruct_2.duration);
        routineTimer.reset();
        }
    
    key_instruct_2.stop();
    // the Routine "break_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


function importConditions(currentLoop) {
  return async function () {
    psychoJS.importAttributes(currentLoop.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}


async function quitPsychoJS(message, isCompleted) {
  // Check for and save orphaned data
  if (psychoJS.experiment.isEntryEmpty()) {
    psychoJS.experiment.nextEntry();
  }
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});
  
  return Scheduler.Event.QUIT;
}
