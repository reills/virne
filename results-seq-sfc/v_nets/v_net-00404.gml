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
  id 404
  arrival_time 8021.942092575486
  lifetime 2117.09597446713
  num_nodes 5
  type "path"
  node [
    id 0
    label "0"
    cpu 11
    gpu 10
    rom 31
  ]
  node [
    id 1
    label "1"
    cpu 27
    gpu 22
    rom 2
  ]
  node [
    id 2
    label "2"
    cpu 13
    gpu 49
    rom 6
  ]
  node [
    id 3
    label "3"
    cpu 23
    gpu 26
    rom 12
  ]
  node [
    id 4
    label "4"
    cpu 40
    gpu 48
    rom 40
  ]
  edge [
    source 0
    target 1
    bw 29
  ]
  edge [
    source 1
    target 2
    bw 2
  ]
  edge [
    source 2
    target 3
    bw 10
  ]
  edge [
    source 3
    target 4
    bw 19
  ]
]
