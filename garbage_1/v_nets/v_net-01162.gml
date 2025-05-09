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
  id 1162
  arrival_time 24122.664883390673
  lifetime 804.8725422259863
  num_nodes 7
  type "path"
  node [
    id 0
    label "0"
    cpu 45
    gpu 39
    rom 19
  ]
  node [
    id 1
    label "1"
    cpu 42
    gpu 28
    rom 27
  ]
  node [
    id 2
    label "2"
    cpu 23
    gpu 44
    rom 33
  ]
  node [
    id 3
    label "3"
    cpu 40
    gpu 35
    rom 16
  ]
  node [
    id 4
    label "4"
    cpu 25
    gpu 3
    rom 36
  ]
  node [
    id 5
    label "5"
    cpu 29
    gpu 19
    rom 42
  ]
  node [
    id 6
    label "6"
    cpu 24
    gpu 27
    rom 6
  ]
  edge [
    source 0
    target 1
    bw 17
  ]
  edge [
    source 1
    target 2
    bw 1
  ]
  edge [
    source 2
    target 3
    bw 32
  ]
  edge [
    source 3
    target 4
    bw 42
  ]
  edge [
    source 4
    target 5
    bw 4
  ]
  edge [
    source 5
    target 6
    bw 39
  ]
]
