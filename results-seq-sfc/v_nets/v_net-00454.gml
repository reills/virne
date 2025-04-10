graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 454
  arrival_time 8662.814053437756
  lifetime 1162.6642458955967
  num_nodes 3
  type "path"
  node [
    id 0
    label "0"
    cpu 0
    gpu 36
    rom 50
  ]
  node [
    id 1
    label "1"
    cpu 35
    gpu 34
    rom 10
  ]
  node [
    id 2
    label "2"
    cpu 12
    gpu 23
    rom 0
  ]
  edge [
    source 0
    target 1
    bw 45
  ]
  edge [
    source 1
    target 2
    bw 38
  ]
]
