'use strict';

function initializeMap(bounds) {
  const map = L.map('map').fitBounds(bounds);
  const mapLink = '<a href="">OpenStreetMap</a>';
  L.tileLayer(
      '', {
      attribution: '&copy; ' + mapLink + ' Contributors',
      maxZoom: 19,
      }).addTo(map);
  return map;
}

const fetchJson = (url) => fetch(url).then(response => response.json());

function buildAgentClusters(data) {
  const agentClusters = {};

  data.filter(d => d.json && d.json.location).forEach(d => {
    const id = d.id;

    agentClusters[id] = {
      location: d.json.location,
      nResolved: 0,
      nTotal: 0
    };
  });

  const markers = L.featureGroup();
  Object.values(agentClusters).map((cluster) => {
    const loc = [cluster.location.latitude, cluster.location.longitude];

    L.circleMarker(loc, {
        radius: 10,
        fillColor: 'blue',
        fillOpacity: 0.1,
        stroke: false
      })
      .addTo(markers);
  });
  return markers;
}

function buildWalkerClusters(data) {
  const agentClusters = {};

  data.filter(d => d.json && d.json.location).forEach(d => {
    const id = d.walker_id + '-' + d.walk_time;
    agentClusters[id] = {
      location: d.json.location,
      resolved: d.json.resolved == 1
    };
  });

  const markers = L.featureGroup();
  let count = 0;
  Object.values(agentClusters).map((cluster) => {
    const loc = [cluster.location.latitude, cluster.location.longitude];
    L.circleMarker(loc, {
        radius: 1,
        fillColor: 'blue',
        fillOpacity: 0.8,
        stroke: false
      })
      .addTo(markers);
    count += 1;
  });
  // alert(count);
  return markers;
}

function buildResolvedPaths(data) {
  const paths = {};
  let maxTotal = 1;
  let startTime = 0;
  // let nLines = 0;
  data.filter(d => d.json && d.json.location).forEach(d => {
    const id = d.walker_id;
    if (!paths[id]) paths[id] = [];
    // if (startTime == 0 || d.walk_time < startTime) {
    //   startTime = d.walk_time;
    // } else if (startTime == d.walk_time) {
    //   nLines += 1;
    // }
    paths[id].push({
      location: d.json.location,
      time: d.walk_time,
      nodeId: d.walker_id + '-' + d.walk_time,
      resolved: d.json.resolved == 1
    });
  });

  const lines = L.featureGroup();
  let lineIdx = 0;
  const nLines = Object.keys(paths).length;
  Object.entries(paths).map(([id, points]) => {

    points.sort(function(a,b) {
      return parseInt(a.time) > parseInt(b.time) ? 1 : -1;
    });
    console.log(points);

    let prevNodeId = null;
    const deduplicated = [];
    points.forEach(p => {
      // console.log(p.nodeId);
      if (p.nodeId !== prevNodeId) deduplicated.push(p);
      prevNodeId = p.nodeId;
    });
    // small offset so it's possible to distinguish overlapping edges
    // when zooming in
    const offset = 0.00002 * lineIdx / nLines;
    const coords = points.map(p => [
      p.location.latitude + offset,
      p.location.longitude + offset*0.5
    ]);
    const hue = Math.round(lineIdx / nLines * 360);
    var polyline = L.polyline(coords, {
      color: `hsl(${hue}, 100%, 80%)`,
      opacity: 0.5
    });
    polyline.addTo(lines);
    lineIdx++;

    polyline.on('mouseover', function(ev) {
      this.setStyle({
        color: 'black',   //or whatever style you wish to use;
        opacity: 1
      });
    });
    polyline.on('mouseout', function() {
      this.setStyle({
        color: `hsl(${hue}, 100%, 80%)`,
        opacity: 0.5
      })
    });
    polyline.on('click', function () {
      alert("You clicked the map at " + id);
    });
  });
  return lines;
}

Promise.all([
  fetchJson('/agents'),
  fetchJson('/walks_attached')
]).then(([agentData, walkData]) => {
  const agentClusters = buildAgentClusters(agentData);
  const walkerClusters = buildWalkerClusters(walkData);
  const map = initializeMap(agentClusters.getBounds());
  walkerClusters.addTo(map);
  // agentClusters.addTo(map);
  buildResolvedPaths(walkData).addTo(map);
});
