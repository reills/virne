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
  id 421
  arrival_time 8259.444127688941
  lifetime 1165.2910565261673
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 6
    gpu 1
    rom 13
  ]
  node [
    id 1
    label "1"
    cpu 37
    gpu 5
    rom 27
  ]
  node [
    id 2
    label "2"
    cpu 3
    gpu 34
    rom 49
  ]
  node [
    id 3
    label "3"
    cpu 2
    gpu 5
    rom 25
  ]
  node [
    id 4
    label "4"
    cpu 12
    gpu 25
    rom 35
  ]
  node [
    id 5
    label "5"
    cpu 16
    gpu 1
    rom 27
  ]
  node [
    id 6
    label "6"
    cpu 18
    gpu 13
    rom 23
  ]
  node [
    id 7
    label "7"
    cpu 19
    gpu 15
    rom 22
  ]
  node [
    id 8
    label "8"
    cpu 4
    gpu 36
    rom 3
  ]
  node [
    id 9
    label "9"
    cpu 14
    gpu 41
    rom 20
  ]
  edge [
    source 0
    target 1
    bw 32
  ]
  edge [
    source 1
    target 2
    bw 50
  ]
  edge [
    source 2
    target 3
    bw 1
  ]
  edge [
    source 3
    target 4
    bw 41
  ]
  edge [
    source 4
    target 5
    bw 30
  ]
  edge [
    source 5
    target 6
    bw 30
  ]
  edge [
    source 6
    target 7
    bw 21
  ]
  edge [
    source 7
    target 8
    bw 11
  ]
  edge [
    source 8
    target 9
    bw 4
  ]
]
