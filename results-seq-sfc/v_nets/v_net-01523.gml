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
  id 1523
  arrival_time 33749.83979111142
  lifetime 908.1819723000398
  num_nodes 13
  type "path"
  node [
    id 0
    label "0"
    cpu 46
    gpu 8
    rom 28
  ]
  node [
    id 1
    label "1"
    cpu 30
    gpu 46
    rom 40
  ]
  node [
    id 2
    label "2"
    cpu 21
    gpu 2
    rom 16
  ]
  node [
    id 3
    label "3"
    cpu 29
    gpu 22
    rom 30
  ]
  node [
    id 4
    label "4"
    cpu 43
    gpu 1
    rom 31
  ]
  node [
    id 5
    label "5"
    cpu 40
    gpu 13
    rom 4
  ]
  node [
    id 6
    label "6"
    cpu 37
    gpu 15
    rom 25
  ]
  node [
    id 7
    label "7"
    cpu 0
    gpu 47
    rom 3
  ]
  node [
    id 8
    label "8"
    cpu 26
    gpu 31
    rom 45
  ]
  node [
    id 9
    label "9"
    cpu 4
    gpu 17
    rom 36
  ]
  node [
    id 10
    label "10"
    cpu 47
    gpu 12
    rom 10
  ]
  node [
    id 11
    label "11"
    cpu 47
    gpu 1
    rom 10
  ]
  node [
    id 12
    label "12"
    cpu 33
    gpu 7
    rom 26
  ]
  edge [
    source 0
    target 1
    bw 42
  ]
  edge [
    source 1
    target 2
    bw 19
  ]
  edge [
    source 2
    target 3
    bw 4
  ]
  edge [
    source 3
    target 4
    bw 36
  ]
  edge [
    source 4
    target 5
    bw 10
  ]
  edge [
    source 5
    target 6
    bw 24
  ]
  edge [
    source 6
    target 7
    bw 36
  ]
  edge [
    source 7
    target 8
    bw 11
  ]
  edge [
    source 8
    target 9
    bw 44
  ]
  edge [
    source 9
    target 10
    bw 16
  ]
  edge [
    source 10
    target 11
    bw 12
  ]
  edge [
    source 11
    target 12
    bw 40
  ]
]
