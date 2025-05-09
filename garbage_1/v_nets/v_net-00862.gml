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
  id 862
  arrival_time 18065.639392889
  lifetime 68.25478637602781
  num_nodes 8
  type "path"
  node [
    id 0
    label "0"
    cpu 1
    gpu 24
    rom 40
  ]
  node [
    id 1
    label "1"
    cpu 15
    gpu 26
    rom 17
  ]
  node [
    id 2
    label "2"
    cpu 48
    gpu 5
    rom 4
  ]
  node [
    id 3
    label "3"
    cpu 29
    gpu 31
    rom 29
  ]
  node [
    id 4
    label "4"
    cpu 26
    gpu 0
    rom 33
  ]
  node [
    id 5
    label "5"
    cpu 29
    gpu 43
    rom 1
  ]
  node [
    id 6
    label "6"
    cpu 11
    gpu 44
    rom 12
  ]
  node [
    id 7
    label "7"
    cpu 38
    gpu 0
    rom 12
  ]
  edge [
    source 0
    target 1
    bw 21
  ]
  edge [
    source 1
    target 2
    bw 42
  ]
  edge [
    source 2
    target 3
    bw 28
  ]
  edge [
    source 3
    target 4
    bw 46
  ]
  edge [
    source 4
    target 5
    bw 38
  ]
  edge [
    source 5
    target 6
    bw 4
  ]
  edge [
    source 6
    target 7
    bw 36
  ]
]
