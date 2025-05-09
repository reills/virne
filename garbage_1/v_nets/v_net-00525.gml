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
  id 525
  arrival_time 10052.191845340558
  lifetime 183.45787468685413
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 13
    gpu 12
    rom 43
  ]
  node [
    id 1
    label "1"
    cpu 10
    gpu 0
    rom 20
  ]
  node [
    id 2
    label "2"
    cpu 3
    gpu 14
    rom 40
  ]
  node [
    id 3
    label "3"
    cpu 27
    gpu 2
    rom 40
  ]
  node [
    id 4
    label "4"
    cpu 18
    gpu 49
    rom 33
  ]
  node [
    id 5
    label "5"
    cpu 7
    gpu 39
    rom 12
  ]
  node [
    id 6
    label "6"
    cpu 15
    gpu 20
    rom 47
  ]
  node [
    id 7
    label "7"
    cpu 22
    gpu 50
    rom 36
  ]
  node [
    id 8
    label "8"
    cpu 34
    gpu 36
    rom 33
  ]
  node [
    id 9
    label "9"
    cpu 12
    gpu 19
    rom 10
  ]
  node [
    id 10
    label "10"
    cpu 29
    gpu 10
    rom 46
  ]
  edge [
    source 0
    target 1
    bw 38
  ]
  edge [
    source 1
    target 2
    bw 44
  ]
  edge [
    source 2
    target 3
    bw 1
  ]
  edge [
    source 3
    target 4
    bw 8
  ]
  edge [
    source 4
    target 5
    bw 18
  ]
  edge [
    source 5
    target 6
    bw 46
  ]
  edge [
    source 6
    target 7
    bw 10
  ]
  edge [
    source 7
    target 8
    bw 42
  ]
  edge [
    source 8
    target 9
    bw 20
  ]
  edge [
    source 9
    target 10
    bw 28
  ]
]
