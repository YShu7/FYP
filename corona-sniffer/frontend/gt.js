'use strict';

function initializeMap(bounds) {
  const map = L.map('map').fitBounds(bounds);
  const mapLink = '<a href="http://openstreetmap.org">OpenStreetMap</a>';
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
  Object.values(agentClusters).map((cluster) => {
    const loc = [cluster.location.latitude, cluster.location.longitude];
    L.circleMarker(loc, {
        radius: 1,
        fillColor: 'blue',
        fillOpacity: 0.8,
        stroke: false
      })
      .addTo(markers);
  });
  
  return markers;
}

function buildResolvedPaths(data) {
  const paths = {};
  let maxTotal = 1;
  data.filter(d => d.json && d.json.location).forEach(d => {
    const id = d.real_id;
    if (!paths[id]) paths[id] = [];
    paths[id].push({
      location: d.json.location,
      time: d.walk_time,
      nodeId: d.real_id,
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

    let prevNodeId = null;
    const deduplicated = [];
    points.forEach(p => {
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
    L.polyline(coords, {
      color: `hsl(${hue}, 100%, 80%)`
    }).addTo(lines);
    lineIdx++;
  });
  return lines;
}

Promise.all([
  fetchJson('/agents'),
  fetchJson('/gt')
]).then(([agentData, walkData]) => {
  const agentClusters = buildAgentClusters(agentData);
  const walkerClusters = buildWalkerClusters(walkData);
  // const map = initializeMap(walkerClusters.getBounds());
  const map = initializeMap(agentClusters.getBounds());
  // agentClusters.addTo(map);
  walkerClusters.addTo(map);
  buildResolvedPaths(walkData).addTo(map);
});
