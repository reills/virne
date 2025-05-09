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
  id 681
  arrival_time 14410.273553771041
  lifetime 2626.1911022211502
  num_nodes 9
  type "path"
  node [
    id 0
    label "0"
    cpu 9
    gpu 28
    rom 34
  ]
  node [
    id 1
    label "1"
    cpu 50
    gpu 29
    rom 39
  ]
  node [
    id 2
    label "2"
    cpu 20
    gpu 40
    rom 45
  ]
  node [
    id 3
    label "3"
    cpu 27
    gpu 19
    rom 2
  ]
  node [
    id 4
    label "4"
    cpu 36
    gpu 12
    rom 33
  ]
  node [
    id 5
    label "5"
    cpu 14
    gpu 7
    rom 33
  ]
  node [
    id 6
    label "6"
    cpu 6
    gpu 9
    rom 3
  ]
  node [
    id 7
    label "7"
    cpu 11
    gpu 45
    rom 7
  ]
  node [
    id 8
    label "8"
    cpu 44
    gpu 27
    rom 33
  ]
  edge [
    source 0
    target 1
    bw 8
  ]
  edge [
    source 1
    target 2
    bw 19
  ]
  edge [
    source 2
    target 3
    bw 24
  ]
  edge [
    source 3
    target 4
    bw 45
  ]
  edge [
    source 4
    target 5
    bw 44
  ]
  edge [
    source 5
    target 6
    bw 18
  ]
  edge [
    source 6
    target 7
    bw 9
  ]
  edge [
    source 7
    target 8
    bw 11
  ]
]
