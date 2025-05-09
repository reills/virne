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
  id 1671
  arrival_time 37286.59596593847
  lifetime 386.02533636732585
  num_nodes 12
  type "path"
  node [
    id 0
    label "0"
    cpu 40
    gpu 28
    rom 10
  ]
  node [
    id 1
    label "1"
    cpu 20
    gpu 35
    rom 15
  ]
  node [
    id 2
    label "2"
    cpu 24
    gpu 9
    rom 37
  ]
  node [
    id 3
    label "3"
    cpu 0
    gpu 33
    rom 47
  ]
  node [
    id 4
    label "4"
    cpu 36
    gpu 30
    rom 7
  ]
  node [
    id 5
    label "5"
    cpu 14
    gpu 3
    rom 30
  ]
  node [
    id 6
    label "6"
    cpu 23
    gpu 11
    rom 34
  ]
  node [
    id 7
    label "7"
    cpu 46
    gpu 37
    rom 32
  ]
  node [
    id 8
    label "8"
    cpu 47
    gpu 46
    rom 43
  ]
  node [
    id 9
    label "9"
    cpu 18
    gpu 16
    rom 22
  ]
  node [
    id 10
    label "10"
    cpu 50
    gpu 8
    rom 9
  ]
  node [
    id 11
    label "11"
    cpu 43
    gpu 15
    rom 13
  ]
  edge [
    source 0
    target 1
    bw 9
  ]
  edge [
    source 1
    target 2
    bw 32
  ]
  edge [
    source 2
    target 3
    bw 17
  ]
  edge [
    source 3
    target 4
    bw 18
  ]
  edge [
    source 4
    target 5
    bw 12
  ]
  edge [
    source 5
    target 6
    bw 49
  ]
  edge [
    source 6
    target 7
    bw 20
  ]
  edge [
    source 7
    target 8
    bw 42
  ]
  edge [
    source 8
    target 9
    bw 44
  ]
  edge [
    source 9
    target 10
    bw 46
  ]
  edge [
    source 10
    target 11
    bw 13
  ]
]
